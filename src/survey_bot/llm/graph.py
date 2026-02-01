"""
LangGraph state machine for survey navigation.

This module implements the core navigation loop using LangGraph:
1. OBSERVE: Extract page state
2. DECIDE: Use LLM to choose action
3. EXECUTE: Perform the action
4. CHECK: Verify completion or continue

The state machine handles the entire survey flow from start to finish.

Example Usage:
    >>> from survey_bot.llm.graph import create_navigation_graph, SurveyState
    >>> from survey_bot.browser import BrowserManager, PageObserver, PageInteractor
    >>> 
    >>> # Create the graph
    >>> graph = create_navigation_graph(persona, llm, observer, interactor)
    >>> 
    >>> # Run the survey
    >>> async with BrowserManager() as page:
    ...     initial_state = SurveyState(page=page, persona=persona, ...)
    ...     final_state = await graph.ainvoke(initial_state)
    ...     print(f"Coupon: {final_state['coupon_code']}")
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Literal, TypedDict, Optional, List

# LangGraph imports - handle gracefully if not installed
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"

from ..models.persona import PersonaConfig, MoodType
from ..models.page_state import PageState, QuestionType, WebElement
from ..browser.observer import PageObserver
from ..browser.interactor import PageInteractor
from .chains import (
    ActionType,
    NavigationAction,
    NavigationDecision,
    make_decision,
    get_llm,
)


__all__ = [
    "SurveyState",
    "create_navigation_graph",
    "NavigationGraph",
    "run_survey_navigation",
]

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SurveyState(TypedDict, total=False):
    """
    State for the survey navigation graph.
    
    This TypedDict defines all the information tracked during
    survey navigation.
    """
    # Page reference (not serializable, passed separately)
    page_url: str
    page_title: str
    
    # Current page state from observer
    page_state: Optional[dict]  # Serialized PageState
    
    # Persona configuration
    persona: dict  # Serialized PersonaConfig
    mood: str  # MoodType value
    
    # Navigation tracking
    actions_taken: list[str]
    current_step: int
    max_steps: int
    
    # Completion status
    is_complete: bool
    coupon_code: Optional[str]
    error: Optional[str]
    
    # Current decision
    current_action: Optional[dict]  # Serialized NavigationAction
    
    # Timing
    start_time: str
    last_action_time: str


# =============================================================================
# DEFAULT STATE
# =============================================================================

def create_initial_state(
    persona: PersonaConfig,
    max_steps: int = 50,
) -> SurveyState:
    """
    Create initial state for the navigation graph.
    
    Args:
        persona: The persona configuration.
        max_steps: Maximum steps before timeout.
        
    Returns:
        Initial SurveyState dictionary.
    """
    return SurveyState(
        page_url="",
        page_title="",
        page_state=None,
        persona=persona.model_dump(),
        mood=persona.mood.value,
        actions_taken=[],
        current_step=0,
        max_steps=max_steps,
        is_complete=False,
        coupon_code=None,
        error=None,
        current_action=None,
        start_time=datetime.now().isoformat(),
        last_action_time=datetime.now().isoformat(),
    )


# =============================================================================
# NAVIGATION GRAPH CLASS
# =============================================================================

class NavigationGraph:
    """
    Encapsulates the LangGraph navigation state machine.
    
    Provides a cleaner interface for running the navigation loop
    with proper resource management.
    
    Attributes:
        persona: The persona configuration.
        observer: PageObserver instance.
        interactor: PageInteractor instance.
        llm: LangChain LLM instance (optional).
        max_steps: Maximum navigation steps.
    """
    
    def __init__(
        self,
        persona: PersonaConfig,
        observer: Optional[PageObserver] = None,
        interactor: Optional[PageInteractor] = None,
        llm: Optional[Any] = None,
        max_steps: int = 50,
    ):
        """
        Initialize the NavigationGraph.
        
        Args:
            persona: Persona configuration for responses.
            observer: PageObserver instance (created if None).
            interactor: PageInteractor instance (created if None).
            llm: LangChain LLM for decisions (uses rules if None).
            max_steps: Maximum steps before timeout.
        """
        self.persona = persona
        self.observer = observer or PageObserver()
        self.interactor = interactor or PageInteractor()
        self.llm = llm
        self.max_steps = max_steps
        self._user_email: Optional[str] = None  # Will be set during run()
        
        # Build the graph
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
        
        logger.info(
            f"NavigationGraph initialized: persona={persona.mood.value}, "
            f"max_steps={max_steps}, llm={'yes' if llm else 'no'}"
        )
    
    def _build_graph(self):
        """Build the LangGraph state machine."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not installed. Run: pip install langgraph"
            )
        
        # Create the graph with state schema
        builder = StateGraph(SurveyState)
        
        # Add nodes
        builder.add_node("observe_page", self._observe_page)
        builder.add_node("decide_action", self._decide_action)
        builder.add_node("execute_action", self._execute_action)
        builder.add_node("check_completion", self._check_completion)
        
        # Add edges
        builder.set_entry_point("observe_page")
        builder.add_edge("observe_page", "decide_action")
        builder.add_edge("decide_action", "execute_action")
        builder.add_edge("execute_action", "check_completion")
        
        # Conditional edge from check_completion
        builder.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "observe_page",
                "end": END,
            }
        )
        
        # Compile the graph
        return builder.compile()
    
    async def _observe_page(self, state: SurveyState, page) -> Dict[str, Any]:
        """
        NODE: Observe the current page state.
        
        Calls PageObserver to extract all interactive elements
        and detect the question type.
        """
        logger.debug(f"Step {state['current_step']}: Observing page...")
        
        try:
            # Get page state
            page_state = await self.observer.get_page_state(page)
            
            # Check for coupon immediately
            if page_state.has_coupon:
                logger.info(f"Coupon detected: {page_state.coupon_code}")
                return {
                    "page_state": page_state.model_dump(),
                    "page_url": page_state.url,
                    "page_title": page_state.title,
                    "is_complete": True,
                    "coupon_code": page_state.coupon_code,
                }
            
            return {
                "page_state": page_state.model_dump(),
                "page_url": page_state.url,
                "page_title": page_state.title,
            }
            
        except Exception as e:
            logger.error(f"Observation failed: {e}")
            return {"error": str(e)}
    
    async def _decide_action(self, state: SurveyState) -> Dict[str, Any]:
        """
        NODE: Decide what action to take.
        
        Uses LLM or rules to determine the next action
        based on current page state and persona.
        """
        logger.debug(f"Step {state['current_step']}: Deciding action...")
        
        # If already complete, skip decision
        if state.get("is_complete"):
            return {
                "current_action": NavigationAction(
                    action_type=ActionType.DONE,
                    reasoning="Survey already complete",
                ).model_dump()
            }
        
        # Reconstruct PageState from dict
        page_state_dict = state.get("page_state")
        if not page_state_dict:
            return {
                "current_action": NavigationAction(
                    action_type=ActionType.WAIT,
                    reasoning="No page state available",
                ).model_dump()
            }
        
        # Use model_validate for proper nested model conversion
        try:
            page_state = PageState.model_validate(page_state_dict)
        except Exception as e:
            logger.warning(f"Failed to validate PageState, using raw dict: {e}")
            page_state = PageState(**page_state_dict)
        
        # Make decision
        decision = await make_decision(
            llm=self.llm,
            persona=self.persona,
            page_state=page_state,
            actions_taken=state.get("actions_taken", []),
        )
        
        logger.info(f"Decision: {decision.action.action_type} - {decision.action.reasoning}")
        
        # Update state if complete
        updates: Dict[str, Any] = {
            "current_action": decision.action.model_dump(),
        }

        if decision.is_complete:
            updates["is_complete"] = True
            if decision.detected_coupon:
                updates["coupon_code"] = decision.detected_coupon
        
        return updates
    
    async def _execute_action(self, state: SurveyState, page) -> Dict[str, Any]:
        """
        NODE: Execute the decided action.
        
        Calls the appropriate PageInteractor method based on
        the action type.
        """
        action_dict = state.get("current_action")
        if not action_dict:
            return {"error": "No action to execute"}
        
        action = NavigationAction(**action_dict)
        logger.debug(f"Step {state['current_step']}: Executing {action.action_type}...")
        
        # Track action
        action_desc = f"[{state['current_step']}] {action.action_type}"
        if action.value:
            action_desc += f": {str(action.value)[:30]}"
        
        try:
            success = await self._perform_action(action, state, page)
            
            if success:
                action_desc += " [OK]"
            else:
                action_desc += " [FAIL]"
            
            # Wait for page to settle
            await asyncio.sleep(0.5)
            
            return {
                "actions_taken": state.get("actions_taken", []) + [action_desc],
                "current_step": state.get("current_step", 0) + 1,
                "last_action_time": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action_desc += f" ERROR: {e}"
            return {
                "actions_taken": state.get("actions_taken", []) + [action_desc],
                "current_step": state.get("current_step", 0) + 1,
                "error": str(e),
            }
    
    async def _perform_action(
        self,
        action: NavigationAction,
        state: SurveyState,
        page,
    ) -> bool:
        """Execute a specific action using the interactor."""
        
        if action.action_type == ActionType.DONE:
            return True
        
        if action.action_type == ActionType.WAIT:
            await asyncio.sleep(1)
            return True
        
        # Get page state for element lookup
        page_state_dict = state.get("page_state") or {}
        elements = page_state_dict.get("elements", [])
        
        if action.action_type == ActionType.CLICK:
            # Find target element
            element = self._find_element(elements, action.target_element_id)
            if element:
                web_element = WebElement(**element)
                return await self.interactor.click_element(page, web_element)
            else:
                # Try to find submit button
                submit_btn = self._find_submit_in_elements(elements)
                if submit_btn:
                    web_element = WebElement(**submit_btn)
                    return await self.interactor.click_element(page, web_element)
            return False
        
        if action.action_type == ActionType.FILL_TEXT:
            element = self._find_element(elements, action.target_element_id)
            if element:
                web_element = WebElement(**element)
                return await self.interactor.fill_text(
                    page, web_element, action.value or ""
                )
            # Try first text input
            for elem in elements:
                elem_type = elem.get("element_type")
                # Handle both string and enum element types
                if elem_type:
                    if hasattr(elem_type, 'value'):
                        elem_type_str = elem_type.value
                    else:
                        elem_type_str = str(elem_type)
                    
                    if elem_type_str in ("text_input", "textarea"):
                        web_element = WebElement(**elem)
                        return await self.interactor.fill_text(
                            page, web_element, action.value or ""
                        )
            return False
        
        if action.action_type == ActionType.SELECT_RATING:
            rating = int(action.value or self.persona.rating_preference)
            return await self.interactor.select_rating(page, rating)
        
        if action.action_type == ActionType.SELECT_OPTION:
            return await self.interactor.select_option(
                page, action.value or ""
            )
        
        return False
    
    def _find_element(
        self,
        elements: list[dict],
        element_id: Optional[str],
    ) -> Optional[dict]:
        """Find element by ID in elements list."""
        if not element_id:
            return None
        
        for elem in elements:
            if elem.get("element_id") == element_id:
                return elem
        
        return None
    
    def _find_submit_in_elements(self, elements: list[dict]) -> Optional[dict]:
        """Find a submit/next button in elements."""
        submit_keywords = ["next", "continue", "submit", "proceed", "finish"]
        
        for elem in elements:
            # Handle both string and enum element types
            elem_type = elem.get("element_type")
            if elem_type:
                # Convert to string if it's an enum
                if hasattr(elem_type, 'value'):
                    elem_type_str = elem_type.value
                else:
                    elem_type_str = str(elem_type)
                
                if elem_type_str in ("button", "link"):
                    text = (elem.get("text_content") or "").lower()
                    if any(kw in text for kw in submit_keywords):
                        return elem
        
        return None
    
    async def _check_completion(self, state: SurveyState) -> dict:
        """
        NODE: Check if survey is complete.
        
        Determines whether to continue the loop or end.
        """
        # Already complete?
        if state.get("is_complete"):
            logger.info("Survey complete!")
            return {}
        
        # Error occurred?
        if state.get("error"):
            logger.error(f"Survey error: {state['error']}")
            return {"is_complete": True}
        
        # Max steps reached?
        if state.get("current_step", 0) >= state.get("max_steps", 50):
            logger.warning("Max steps reached")
            return {
                "is_complete": True,
                "error": "Maximum steps reached without completion",
            }
        
        # Coupon found?
        if state.get("coupon_code"):
            logger.info(f"Coupon found: {state['coupon_code']}")
            return {"is_complete": True}
        
        return {}
    
    def _should_continue(self, state: SurveyState) -> Literal["continue", "end"]:
        """Determine if the loop should continue."""
        if state.get("is_complete") or state.get("error"):
            return "end"
        return "continue"
    
    async def run(self, page, initial_url: str, survey_code: Optional[str] = None, email: Optional[str] = None) -> SurveyState:
        """
        Run the navigation loop on a page.
        
        Args:
            page: Playwright Page object.
            initial_url: Starting URL for the survey.
            survey_code: Survey validation code from receipt (optional).
            email: Email address for coupon delivery (optional).
            
        Returns:
            Final SurveyState with results.
        """
        # Store email for use during navigation
        self._user_email = email
        
        # Navigate to initial URL
        await page.goto(initial_url, wait_until="networkidle")
        
        # Wait for page to be interactive - try to find any clickable element
        try:
            await page.wait_for_selector(
                "button, input, a[href], [role='button']",
                timeout=10000
            )
            logger.info("Page has interactive elements")
        except Exception as e:
            logger.warning(f"No interactive elements found after 10s: {e}")
            # Try waiting a bit more for dynamic content
            await asyncio.sleep(2)
        
        # Log page info for debugging
        current_url = page.url
        page_title = await page.title()
        logger.info(f"Current URL: {current_url}")
        logger.info(f"Page title: {page_title}")
        
        # Debug: Save screenshot if page seems empty or redirected
        if current_url != initial_url or not page_title:
            try:
                from pathlib import Path
                debug_dir = Path("screenshots")
                debug_dir.mkdir(exist_ok=True)
                screenshot_path = debug_dir / f"debug_{datetime.now().strftime('%H%M%S')}.png"
                await page.screenshot(path=str(screenshot_path))
                logger.info(f"Debug screenshot saved: {screenshot_path}")
                
                # Also log page HTML snippet
                html_content = await page.content()
                logger.info(f"Page HTML length: {len(html_content)} chars")
                logger.debug(f"Page HTML snippet: {html_content[:500]}")
            except Exception as e:
                logger.warning(f"Failed to save debug screenshot: {e}")
        
        # =================================================================
        # SURVEY CODE ENTRY - Handle initial code entry page
        # =================================================================
        if survey_code:
            logger.info(f"Attempting to fill survey code: {survey_code}")
            try:
                # Check if this looks like a code entry page
                page_text = await page.inner_text("body")
                page_lower = page_text.lower()
                
                # Common indicators of code entry page
                is_code_page = any(phrase in page_lower for phrase in [
                    "survey code",
                    "enter the",
                    "digit code",
                    "validation code",
                    "receipt code",
                    "24-digit",
                ])
                
                if is_code_page:
                    logger.info("Detected survey code entry page")
                    
                    # Fill the survey code
                    code_filled = await self.interactor.fill_survey_code(page, survey_code)
                    
                    if code_filled:
                        logger.info("Survey code entered successfully")
                        
                        # Look for Start/Begin/Submit button
                        start_button = page.locator(
                            "button:has-text('Start'), "
                            "button:has-text('Begin'), "
                            "button:has-text('Submit'), "
                            "button:has-text('Next'), "
                            "input[type='submit'], "
                            "input[value*='Start' i], "
                            "input[value*='Begin' i]"
                        ).first
                        
                        if await start_button.is_visible():
                            logger.info("Clicking Start button")
                            await start_button.click()
                            
                            # Wait for navigation
                            await page.wait_for_load_state("networkidle")
                            await asyncio.sleep(1)
                            
                            logger.info(f"After code entry, URL: {page.url}")
                            
                            # Check for error messages (code already used, invalid, etc.)
                            error_text = await page.inner_text("body")
                            error_lower = error_text.lower()
                            
                            ERROR_PHRASES = [
                                "already been used", "already used", "already completed",
                                "invalid code", "code is invalid", "code not found",
                                "expired", "no longer valid", "not valid",
                                "unable to process", "code has been redeemed",
                                "previously used", "error",
                            ]
                            
                            for error_phrase in ERROR_PHRASES:
                                if error_phrase in error_lower:
                                    logger.error(f"Survey code error detected: '{error_phrase}'")
                                    # Return early with error state
                                    state = create_initial_state(self.persona, self.max_steps)
                                    state["error"] = f"Survey code error: {error_phrase}"
                                    state["is_complete"] = False
                                    return state
                        else:
                            logger.warning("Could not find Start button after code entry")
                    else:
                        logger.warning("Failed to fill survey code")
                else:
                    logger.info("Not a code entry page, proceeding with survey")
                    
            except Exception as e:
                logger.error(f"Error during survey code entry: {e}")
        
        # Create initial state
        state = create_initial_state(self.persona, self.max_steps)
        
        # Run the loop manually (since graph needs page reference)
        while not state.get("is_complete") and state.get("current_step", 0) < self.max_steps:
            # Observe
            updates = await self._observe_page(state, page)
            state.update(updates)  # type: ignore[typeddict-item]
            
            if state.get("is_complete"):
                break
            
            # Auto-fill email if we have one and there's an email field
            if self._user_email:
                await self._try_fill_email(page)
            
            # Decide
            updates = await self._decide_action(state)
            state.update(updates)  # type: ignore[typeddict-item]
            
            if state.get("is_complete"):
                break
            
            # Execute
            updates = await self._execute_action(state, page)
            state.update(updates)  # type: ignore[typeddict-item]
            
            # Check
            updates = await self._check_completion(state)
            state.update(updates)  # type: ignore[typeddict-item]
            
            # Small delay between iterations
            await asyncio.sleep(0.3)
        
        return state
    
    async def _try_fill_email(self, page) -> bool:
        """
        Try to fill email field if present on page.
        
        Looks for common email input patterns and fills with user's email.
        """
        try:
            # Common email field selectors
            email_selectors = [
                "input[type='email']",
                "input[name*='email' i]",
                "input[id*='email' i]",
                "input[placeholder*='email' i]",
                "input[name*='mail' i]",
                "input[autocomplete='email']",
            ]
            
            for selector in email_selectors:
                try:
                    email_input = page.locator(selector).first
                    if await email_input.is_visible():
                        # Check if already filled
                        current_value = await email_input.input_value()
                        if not current_value:
                            logger.info(f"Found email field, filling with: {self._user_email}")
                            await email_input.click()
                            await email_input.fill(self._user_email)
                            return True
                except Exception:
                    continue
            
            return False
            
        except Exception as e:
            logger.debug(f"Email fill attempt: {e}")
            return False


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_navigation_graph(
    persona: PersonaConfig,
    llm: Optional[Any] = None,
    observer: Optional[PageObserver] = None,
    interactor: Optional[PageInteractor] = None,
    max_steps: int = 50,
) -> NavigationGraph:
    """
    Create a navigation graph with all dependencies.
    
    Args:
        persona: Persona configuration.
        llm: LangChain LLM instance (optional).
        observer: PageObserver instance (optional).
        interactor: PageInteractor instance (optional).
        max_steps: Maximum navigation steps.
        
    Returns:
        Configured NavigationGraph instance.
    """
    return NavigationGraph(
        persona=persona,
        observer=observer,
        interactor=interactor,
        llm=llm,
        max_steps=max_steps,
    )


async def run_survey_navigation(
    page,
    survey_url: str,
    persona: PersonaConfig,
    llm: Optional[Any] = None,
    max_steps: int = 50,
    survey_code: Optional[str] = None,
    email: Optional[str] = None,
) -> SurveyState:
    """
    Run complete survey navigation.
    
    Convenience function that creates graph and runs it.
    
    Args:
        page: Playwright Page object.
        survey_url: URL of the survey.
        persona: Persona configuration.
        llm: LangChain LLM instance (optional).
        max_steps: Maximum navigation steps.
        survey_code: Survey validation code from receipt (optional).
        email: Email address for coupon delivery (filled into survey form).
        
    Returns:
        Final SurveyState with results.
    """
    graph = create_navigation_graph(
        persona=persona,
        llm=llm,
        max_steps=max_steps,
    )
    
    return await graph.run(page, survey_url, survey_code=survey_code, email=email)
