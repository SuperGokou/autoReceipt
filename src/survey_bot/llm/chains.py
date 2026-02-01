"""
LangChain chains for survey navigation decision making.

This module provides LangChain-based chains that use LLMs to decide
which actions to take when navigating surveys.

Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Structured output via Pydantic models
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Optional, Dict, List

from pydantic import BaseModel, Field

# LangChain imports - handle gracefully if not installed
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    ChatPromptTemplate = None
    PydanticOutputParser = None

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None


from ..models.persona import PersonaConfig, MoodType
from ..models.page_state import PageState, QuestionType
from .prompts import build_navigator_prompt, build_action_prompt, get_sample_text_response


__all__ = [
    "ActionType",
    "NavigationAction",
    "NavigationDecision",
    "create_decision_chain",
    "get_llm",
    "make_decision",
]

logger = logging.getLogger(__name__)


# =============================================================================
# ACTION MODELS (Structured Output)
# =============================================================================

class ActionType(str, Enum):
    """Types of actions the navigator can take."""
    CLICK = "click"
    FILL_TEXT = "fill_text"
    SELECT_RATING = "select_rating"
    SELECT_OPTION = "select_option"
    DONE = "done"
    WAIT = "wait"
    ERROR = "error"


class NavigationAction(BaseModel):
    """
    Structured action decision from the LLM.
    
    This model defines what action should be taken on the current page.
    """
    action_type: ActionType = Field(
        description="Type of action to perform"
    )
    target_element_id: Optional[str] = Field(
        default=None,
        description="ID of the element to interact with (for CLICK, FILL_TEXT)"
    )
    value: Optional[str] = Field(
        default=None,
        description="Value to enter (for FILL_TEXT, SELECT_RATING, SELECT_OPTION)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this action was chosen"
    )
    
    model_config = {"use_enum_values": True}


class NavigationDecision(BaseModel):
    """
    Complete navigation decision including action and metadata.
    """
    action: NavigationAction
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision (0-1)"
    )
    detected_coupon: Optional[str] = Field(
        default=None,
        description="Coupon code if detected on page"
    )
    is_complete: bool = Field(
        default=False,
        description="Whether the survey appears complete"
    )


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def get_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs: Any,
):
    """
    Get an LLM instance for decision making.
    
    Args:
        provider: LLM provider ('openai' or 'anthropic').
        model: Model name (defaults based on provider).
        api_key: API key (defaults to environment variable).
        temperature: Sampling temperature (lower = more deterministic).
        **kwargs: Additional arguments for the LLM.
        
    Returns:
        LangChain LLM instance.
        
    Raises:
        ImportError: If required packages are not installed.
        ValueError: If provider is not supported.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain not installed. Run: pip install langchain langchain-openai"
        )
    
    if provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        model = model or "gpt-4o-mini"  # Cost-effective default
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
    
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic not installed. Run: pip install langchain-anthropic"
            )
        
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")
        
        model = model or "claude-3-haiku-20240307"  # Cost-effective default
        
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")


# =============================================================================
# DECISION CHAIN
# =============================================================================

def create_decision_chain(
    llm,
    persona: PersonaConfig,
):
    """
    Create a LangChain chain for navigation decisions.
    
    Args:
        llm: LangChain LLM instance.
        persona: The persona configuration.
        
    Returns:
        Configured chain for making decisions.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not installed")
    
    # Build system prompt with persona
    system_prompt = build_navigator_prompt(persona)
    
    # Create output parser for structured output
    parser = PydanticOutputParser(pydantic_object=NavigationAction)
    
    # Build prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{format_instructions}"),
        ("human", "{action_prompt}"),
    ])
    
    # Create chain
    chain = prompt | llm | parser
    
    return chain, parser


async def make_decision(
    llm,
    persona: PersonaConfig,
    page_state: PageState,
    actions_taken: list[str],
) -> NavigationDecision:
    """
    Make a navigation decision for the current page state.
    
    This is the main function for deciding what action to take.
    Handles both LLM-based decisions and fallback logic.
    
    Args:
        llm: LangChain LLM instance (or None for rule-based).
        persona: The persona configuration.
        page_state: Current page state from PageObserver.
        actions_taken: List of previous actions.
        
    Returns:
        NavigationDecision with action details.
    """
    # Check if survey is complete (coupon detected)
    if page_state.has_coupon:
        logger.info("Coupon detected - survey complete!")
        return NavigationDecision(
            action=NavigationAction(
                action_type=ActionType.DONE,
                reasoning="Coupon/validation code detected on page",
            ),
            detected_coupon=page_state.coupon_code,
            is_complete=True,
            confidence=1.0,
        )
    
    # Try LLM-based decision if available
    if llm is not None and LANGCHAIN_AVAILABLE:
        try:
            return await _make_llm_decision(llm, persona, page_state, actions_taken)
        except Exception as e:
            logger.warning(f"LLM decision failed: {e}, falling back to rules")
    
    # Fallback to rule-based decision
    return _make_rule_based_decision(persona, page_state, actions_taken)


async def _make_llm_decision(
    llm,
    persona: PersonaConfig,
    page_state: PageState,
    actions_taken: list[str],
) -> NavigationDecision:
    """Make decision using LLM."""
    # Build the action prompt
    action_prompt = build_action_prompt(
        page_url=page_state.url,
        page_title=page_state.title,
        question_type=page_state.question_type.value,
        question_text=page_state.question_text,
        elements=page_state.get_visible_elements()[:15],  # Limit elements
        actions_taken=actions_taken,
        mood=persona.mood.value,
    )
    
    # Create chain
    chain, parser = create_decision_chain(llm, persona)
    
    # Invoke chain
    result = await chain.ainvoke({
        "action_prompt": action_prompt,
        "format_instructions": parser.get_format_instructions(),
    })
    
    logger.debug(f"LLM decision: {result.action_type} - {result.reasoning}")
    
    return NavigationDecision(
        action=result,
        confidence=0.85,
        is_complete=result.action_type == ActionType.DONE,
    )


def _make_rule_based_decision(
    persona: PersonaConfig,
    page_state: PageState,
    actions_taken: list[str],
) -> NavigationDecision:
    """
    Make decision using rule-based logic (no LLM).
    
    This is a fallback when LLM is unavailable or fails.
    
    Logic flow:
    1. Find all unanswered questions on the page
    2. Answer the FIRST unanswered question
    3. Only click Submit when ALL questions are answered
    """
    logger.debug(f"Rule-based decision for question type: {page_state.question_type}")
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def get_radio_groups() -> Dict[str, Dict[str, Any]]:
        """Get all radio button groups and their selection status."""
        groups: Dict[str, Dict[str, Any]] = {}
        for elem in page_state.elements:
            # Handle both WebElement objects and dicts
            if isinstance(elem, dict):
                elem_type = elem.get('element_type')
                if hasattr(elem_type, 'value'):
                    elem_type_str = elem_type.value
                else:
                    elem_type_str = str(elem_type)
                
                if elem_type_str == 'radio':
                    # Get group name from selector or value
                    selector = elem.get('selector', '')
                    # Extract name from selector like "[name='rating']"
                    import re
                    match = re.search(r"name='([^']+)'", selector)
                    group_name = match.group(1) if match else selector
                    
                    if group_name not in groups:
                        groups[group_name] = {'elements': [], 'has_selection': False}
                    groups[group_name]['elements'].append(elem)
                    if elem.get('is_checked'):
                        groups[group_name]['has_selection'] = True
            else:
                elem_type = getattr(elem, 'element_type', None)
                if elem_type:
                    elem_type_str = elem_type.value if hasattr(elem_type, 'value') else str(elem_type)
                    if elem_type_str == 'radio':
                        selector = getattr(elem, 'selector', '')
                        import re
                        match = re.search(r"name='([^']+)'", selector)
                        group_name = match.group(1) if match else selector
                        
                        if group_name not in groups:
                            groups[group_name] = {'elements': [], 'has_selection': False}
                        groups[group_name]['elements'].append(elem)
                        if getattr(elem, 'is_checked', False):
                            groups[group_name]['has_selection'] = True
        return groups
    
    def get_unfilled_textareas() -> list:
        """Get textareas that haven't been filled."""
        # CRITICAL: Check if we've already filled a textarea in this session
        # The observer may not capture the typed value, so we track via actions
        for action in actions_taken:
            if 'fill_text' in action.lower():
                logger.debug("Already filled textarea in this session, skipping")
                return []  # Already filled, don't try again!
        
        unfilled = []
        for elem in page_state.elements:
            if isinstance(elem, dict):
                elem_type = elem.get('element_type')
                if hasattr(elem_type, 'value'):
                    elem_type_str = elem_type.value
                else:
                    elem_type_str = str(elem_type)
                
                if elem_type_str == 'textarea':
                    # Check if has content (textarea value or text_content)
                    has_content = elem.get('value') or elem.get('text_content')
                    if not has_content:
                        unfilled.append(elem)
            else:
                elem_type = getattr(elem, 'element_type', None)
                if elem_type:
                    elem_type_str = elem_type.value if hasattr(elem_type, 'value') else str(elem_type)
                    if elem_type_str == 'textarea':
                        has_content = getattr(elem, 'value', None) or getattr(elem, 'text_content', None)
                        if not has_content:
                            unfilled.append(elem)
        return unfilled
    
    def find_submit_button() -> Optional[NavigationDecision]:
        """Find and return action to click submit button."""
        submit_btn = page_state.find_submit_button()
        if submit_btn:
            logger.debug(f"Found submit button: {submit_btn.element_id}")
            return NavigationDecision(
                action=NavigationAction(
                    action_type=ActionType.CLICK,
                    target_element_id=submit_btn.element_id,
                    reasoning="All questions answered, clicking Submit",
                ),
                confidence=0.95,
            )
        return None
    
    def select_radio_option(group_name: str, elements: list, preferred_value: str) -> Optional[NavigationDecision]:
        """Select a radio option from a group."""
        # Try to find option matching preferred value (fuzzy match)
        from difflib import SequenceMatcher

        best_match = None
        best_score = 0.0
        
        for elem in elements:
            if isinstance(elem, dict):
                elem_value = str(elem.get('value', '')).lower()
                elem_id = elem.get('element_id')
            else:
                elem_value = str(getattr(elem, 'value', '')).lower()
                elem_id = getattr(elem, 'element_id', None)
            
            # Check for exact match or fuzzy match
            preferred_lower = preferred_value.lower().replace('_', ' ')
            score = SequenceMatcher(None, preferred_lower, elem_value.replace('_', ' ')).ratio()
            
            if score > best_score:
                best_score = score
                best_match = elem_id
        
        # If no good match, just pick the first option (for happy persona) or last (for angry)
        if best_score < 0.3:
            if persona.mood.value == 'happy':
                # Pick first option (usually most positive)
                best_match = elements[0].get('element_id') if isinstance(elements[0], dict) else elements[0].element_id
            else:
                # Pick last option (usually most negative)
                best_match = elements[-1].get('element_id') if isinstance(elements[-1], dict) else elements[-1].element_id
        
        if best_match:
            return NavigationDecision(
                action=NavigationAction(
                    action_type=ActionType.CLICK,
                    target_element_id=best_match,
                    reasoning=f"Selecting option in '{group_name}' group for {persona.mood.value} persona",
                ),
                confidence=0.85,
            )
        return None
    
    # =========================================================================
    # MAIN DECISION LOGIC
    # =========================================================================
    
    # Get all radio groups and their status
    radio_groups = get_radio_groups()
    logger.debug(f"Found {len(radio_groups)} radio groups: {list(radio_groups.keys())}")
    
    # Find unanswered radio groups
    unanswered_groups = {
        name: data for name, data in radio_groups.items() 
        if not data['has_selection']
    }
    
    if unanswered_groups:
        # Answer the first unanswered group
        group_name, group_data = next(iter(unanswered_groups.items()))
        logger.info(f"Answering unanswered radio group: {group_name}")
        
        # Determine preferred value based on group name and persona
        preferred_values = {
            'rating': str(persona.rating_preference),
            'staff_satisfaction': persona.sample_responses.get('satisfaction', 'Very Satisfied'),
            'recommend': persona.sample_responses.get('yes_no_positive', 'Yes'),
            'satisfaction': persona.sample_responses.get('satisfaction', 'Very Satisfied'),
            'experience': persona.sample_responses.get('experience', 'Excellent'),
        }
        
        # Check if this is a numeric rating (1-10)
        if group_name in ('rating', 'overall_rating', 'score'):
            return NavigationDecision(
                action=NavigationAction(
                    action_type=ActionType.SELECT_RATING,
                    value=str(persona.rating_preference),
                    reasoning=f"Selecting rating {persona.rating_preference} for {group_name}",
                ),
                confidence=0.9,
            )
        
        # For text-based options, use SELECT_OPTION or CLICK
        preferred = preferred_values.get(group_name, 'satisfied' if persona.mood.value == 'happy' else 'dissatisfied')
        result = select_radio_option(group_name, group_data['elements'], preferred)
        if result:
            return result
    
    # Check for unfilled textareas (optional but nice to fill)
    unfilled_textareas = get_unfilled_textareas()
    if unfilled_textareas:
        textarea = unfilled_textareas[0]
        elem_id = textarea.get('element_id') if isinstance(textarea, dict) else textarea.element_id
        
        logger.info(f"Filling textarea: {elem_id}")
        text_response = get_sample_text_response(persona.mood, "medium")
        
        return NavigationDecision(
            action=NavigationAction(
                action_type=ActionType.FILL_TEXT,
                target_element_id=elem_id,
                value=text_response,
                reasoning="Filling comment textarea with persona-appropriate response",
            ),
            confidence=0.85,
        )
    
    # All questions answered - click Submit
    logger.info("All questions answered, looking for Submit button")
    result = find_submit_button()
    if result:
        return result
    
    # Check for completion page
    if page_state.question_type == QuestionType.COMPLETION:
        return NavigationDecision(
            action=NavigationAction(
                action_type=ActionType.DONE,
                reasoning="Survey appears complete",
            ),
            is_complete=True,
            confidence=0.8,
        )
    
    # Fallback: just try clicking submit anyway
    logger.warning("No clear action, attempting to find any navigation button")
    submit_btn = page_state.find_submit_button()
    if submit_btn:
        return NavigationDecision(
            action=NavigationAction(
                action_type=ActionType.CLICK,
                target_element_id=submit_btn.element_id,
                reasoning="Fallback: clicking navigation button",
            ),
            confidence=0.5,
        )
    
    # No clear action - wait
    return NavigationDecision(
        action=NavigationAction(
            action_type=ActionType.WAIT,
            reasoning="No clear action determined, waiting",
        ),
        confidence=0.3,
    )


# =============================================================================
# FALLBACK FOR NO LLM
# =============================================================================

class SimpleLLM:
    """
    Simple rule-based LLM substitute for testing without API keys.
    
    This class mimics the LLM interface but uses rules instead.
    Useful for testing and development.
    """
    
    def __init__(self, persona: PersonaConfig):
        self.persona = persona
    
    async def ainvoke(self, input_dict: dict) -> NavigationAction:
        """Simulate LLM invocation with rules."""
        # Parse the action prompt to understand context
        prompt = input_dict.get("action_prompt", "")
        
        # Simple keyword-based decision
        prompt_lower = prompt.lower()
        
        if "rating" in prompt_lower or "scale" in prompt_lower:
            return NavigationAction(
                action_type=ActionType.SELECT_RATING,
                value=str(self.persona.rating_preference),
                reasoning="Rating question detected",
            )
        
        if "comment" in prompt_lower or "feedback" in prompt_lower:
            return NavigationAction(
                action_type=ActionType.FILL_TEXT,
                value=get_sample_text_response(self.persona.mood),
                reasoning="Text input detected",
            )
        
        if "next" in prompt_lower or "continue" in prompt_lower:
            return NavigationAction(
                action_type=ActionType.CLICK,
                target_element_id="submit",
                reasoning="Navigation button detected",
            )
        
        return NavigationAction(
            action_type=ActionType.WAIT,
            reasoning="No clear action",
        )
