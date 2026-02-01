"""
Vision-based Survey Navigator using Qwen3-VL.

Uses a local vision LLM to analyze screenshots and decide actions,
making the bot flexible enough to handle any survey style.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VisionAction:
    """Action decided by the vision model."""
    action_type: Literal["click", "fill", "select", "done", "wait", "error"]
    target: Optional[str] = None  # Button text, field name, or selector
    value: Optional[str] = None  # Text to fill, option to select
    reasoning: str = ""  # Why this action was chosen


class VisionNavigator:
    """
    Uses Qwen3-VL to analyze survey pages and decide actions.

    Instead of DOM parsing and rules, this navigator:
    1. Takes a screenshot of the page
    2. Sends it to Qwen3-VL with context (mood, email, etc.)
    3. Gets back a specific action to perform
    4. Executes that action
    """

    def __init__(
            self,
            ollama_model: Optional[str] = None,
            ollama_host: Optional[str] = None,
            mood: str = "happy",
            email: Optional[str] = None,
    ):
        self.model = ollama_model or os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl")
        from .ollama_host import detect_ollama_host
        self.host = ollama_host or detect_ollama_host()
        self.mood = mood
        self.email = email

        # Mood to rating mapping
        self.rating_preference = {
            "happy": "highest/best (9-10 out of 10, 5 out of 5, Highly Satisfied, Excellent)",
            "neutral": "middle/average (5-7 out of 10, 3 out of 5, Neutral, Average)",
            "angry": "lowest/worst (1-3 out of 10, 1 out of 5, Very Dissatisfied, Poor)",
        }.get(mood, "highest/best")

        logger.info(f"VisionNavigator initialized: model={self.model}, mood={mood}")

    async def analyze_page(self, page, step: int = 0) -> VisionAction:
        """
        Analyze current page and decide what action to take.

        Args:
            page: Playwright page object
            step: Current step number for logging

        Returns:
            VisionAction with the decided action
        """
        try:
            # Take screenshot
            screenshot_bytes = await page.screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            # Get page URL and title for context
            url = page.url
            title = await page.title()

            # Build prompt
            prompt = self._build_prompt(url, title, step)

            # Call Qwen3-VL
            response = await self._call_vision_llm(prompt, screenshot_b64)

            # Parse response into action
            action = self._parse_response(response)

            logger.info(f"Step {step}: Vision decided: {action.action_type} - {action.target or action.value or ''}")
            logger.debug(f"Reasoning: {action.reasoning}")

            return action

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return VisionAction(
                action_type="error",
                reasoning=str(e)
            )

    def _build_prompt(self, url: str, title: str, step: int) -> str:
        """Build the prompt for vision analysis."""

        prompt = f"""You are a survey completion bot. Analyze this survey page screenshot and decide what action to take.

CONTEXT:
- Survey URL: {url}
- Page title: {title}
- Step: {step}
- User mood: {self.mood} (use {self.rating_preference} ratings)
- User email: {self.email or "not provided"}

INSTRUCTIONS:
1. Look at the screenshot carefully
2. Identify what type of page this is (question, rating, text input, email entry, completion, error)
3. Decide the ONE action to take next

CRITICAL RULES:
- If you see a "Next", "Continue", or "Submit" button is ENABLED (not grayed out), CLICK it to proceed
- If radio buttons/checkboxes are ALREADY SELECTED (filled in), click Next to proceed
- If the page has MULTIPLE questions, select ALL unselected options before clicking Next
- DO NOT keep selecting the same option - if it's already selected, move on

POSSIBLE ACTIONS:
- CLICK: [button text] - Click a button (e.g., "Next", "Submit", "Start", "Continue")
- FILL: [field name] = [value] - Fill a text field
- SELECT: [option text or number] - Select a radio button, checkbox, or rating (ONLY if not already selected)
- DONE - Survey is complete (coupon code visible, thank you page, or validation code shown)
- WAIT - Page is loading or unclear, wait and retry
- ERROR: [message] - Something went wrong (code already used, invalid, etc.)

RATING RULES (based on {self.mood} mood):
- For 1-10 scales: choose {self.rating_preference}
- For 1-5 scales: choose {self.rating_preference}  
- For Satisfied/Dissatisfied: choose based on mood
- For Yes/No questions about positive experience: Yes if happy, No if angry
- For "Would you recommend?": Yes/Definitely if happy, No if angry

EMAIL HANDLING:
- If you see an email field, fill it with: {self.email or "skip if no email provided"}
- If there's a "Confirm Email" field, fill it with the same email

ERROR DETECTION:
- If you see "already used", "invalid", "expired", "error" - respond with ERROR

RESPONSE FORMAT (use exactly this format):
ACTION: [action type]
TARGET: [button text, field name, or option]
VALUE: [text to fill or option to select, if applicable]
REASON: [brief explanation]

Example responses:
ACTION: CLICK
TARGET: Next
REASON: Moving to next question

ACTION: SELECT
TARGET: 10
REASON: Selecting highest rating for happy mood

ACTION: FILL
TARGET: Email Address
VALUE: user@email.com
REASON: Entering email for coupon delivery

ACTION: DONE
REASON: Survey complete, validation code displayed

ACTION: ERROR
TARGET: Code already used
REASON: This survey code has already been redeemed

Now analyze the screenshot and respond with your action:"""

        return prompt

    async def _call_vision_llm(self, prompt: str, image_b64: str) -> str:
        """Call vision model with retry logic. Routes to Ollama or DashScope."""
        from .ollama_host import call_vision_llm

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"Calling vision LLM (attempt {attempt + 1}/{max_retries})...")
                return await call_vision_llm(
                    prompt=prompt,
                    image_b64=image_b64,
                    model=self.model,
                    host=self.host,
                )

            except httpx.ConnectError as e:
                last_error = f"Cannot connect to Ollama at {self.host}. Is it running? Error: {e}"
                logger.warning(f"Connection attempt {attempt + 1} failed: {last_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Wait before retry

            except httpx.TimeoutException as e:
                last_error = f"Ollama request timed out. The model may be loading. Error: {e}"
                logger.warning(f"Timeout on attempt {attempt + 1}: {last_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)  # Wait longer before retry

            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

        raise Exception(last_error or "Failed to connect to Ollama after multiple attempts")

    def _parse_response(self, response: str) -> VisionAction:
        """Parse LLM response into VisionAction."""

        response_upper = response.upper()

        # Extract action type
        action_type = "wait"  # default
        if "ACTION: CLICK" in response_upper:
            action_type = "click"
        elif "ACTION: FILL" in response_upper:
            action_type = "fill"
        elif "ACTION: SELECT" in response_upper:
            action_type = "select"
        elif "ACTION: DONE" in response_upper:
            action_type = "done"
        elif "ACTION: ERROR" in response_upper:
            action_type = "error"
        elif "ACTION: WAIT" in response_upper:
            action_type = "wait"

        # Extract target
        target = None
        target_match = re.search(r'TARGET:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if target_match:
            target = target_match.group(1).strip()

        # Extract value
        value = None
        value_match = re.search(r'VALUE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if value_match:
            value = value_match.group(1).strip()

        # Extract reasoning
        reasoning = ""
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return VisionAction(
            action_type=action_type,
            target=target,
            value=value,
            reasoning=reasoning,
        )

    async def execute_action(self, page, action: VisionAction) -> bool:
        """
        Execute the decided action on the page.

        Args:
            page: Playwright page object
            action: VisionAction to execute

        Returns:
            True if action succeeded, False otherwise
        """
        try:
            if action.action_type == "done":
                logger.info("Survey complete!")
                return True

            if action.action_type == "error":
                logger.error(f"Survey error: {action.target}")
                return False

            if action.action_type == "wait":
                await asyncio.sleep(1)
                return True

            if action.action_type == "click":
                return await self._execute_click(page, action.target)

            if action.action_type == "fill":
                return await self._execute_fill(page, action.target, action.value)

            if action.action_type == "select":
                return await self._execute_select(page, action.target)

            return False

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

    async def _execute_click(self, page, target: str) -> bool:
        """Click a button or link."""
        if not target:
            return False

        # Try multiple strategies to find and click
        strategies = [
            # Exact button text
            f"button:has-text('{target}')",
            f"input[value='{target}' i]",
            f"a:has-text('{target}')",
            # Partial match
            f"button:text-matches('{target}', 'i')",
            f"[role='button']:has-text('{target}')",
            # Common variations
            f"input[type='submit'][value*='{target}' i]",
            f".btn:has-text('{target}')",
        ]

        for selector in strategies:
            try:
                element = page.locator(selector).first
                if await element.is_visible(timeout=1000):
                    await element.click()
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    logger.info(f"Clicked: {target}")
                    return True
            except Exception:
                continue

        # Fallback: try clicking any visible button/link containing the text
        try:
            await page.click(f"text='{target}'", timeout=3000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            logger.info(f"Clicked (fallback): {target}")
            return True
        except Exception:
            pass

        logger.warning(f"Could not find clickable element: {target}")
        return False

    async def _execute_fill(self, page, target: str, value: str) -> bool:
        """Fill a text input field."""
        if not target or not value:
            return False

        target_lower = target.lower()

        # Build selectors based on field name
        selectors = []

        if "email" in target_lower:
            selectors = [
                "input[type='email']",
                "input[name*='email' i]",
                "input[id*='email' i]",
                "input[placeholder*='email' i]",
                f"input[name*='{target}' i]",
            ]
            # Handle both email and confirm email fields
            if "confirm" in target_lower:
                selectors = [
                                "input[name*='confirm' i][name*='email' i]",
                                "input[id*='confirm' i]",
                                "input[placeholder*='confirm' i]",
                            ] + selectors
        else:
            selectors = [
                f"input[name*='{target}' i]",
                f"input[id*='{target}' i]",
                f"input[placeholder*='{target}' i]",
                f"textarea[name*='{target}' i]",
                "input[type='text']",
                "textarea",
            ]

        for selector in selectors:
            try:
                elements = page.locator(selector)
                count = await elements.count()

                for i in range(count):
                    element = elements.nth(i)
                    if await element.is_visible(timeout=1000):
                        current = await element.input_value()
                        if not current:  # Only fill if empty
                            await element.click()
                            await element.fill(value)
                            logger.info(f"Filled '{target}' with '{value}'")
                            return True
            except Exception:
                continue

        logger.warning(f"Could not find field: {target}")
        return False

    async def _execute_select(self, page, target: str) -> bool:
        """Select a radio button, checkbox, or rating option."""
        if not target:
            return False

        # Check if target is a number (rating)
        try:
            rating = int(target)
            return await self._select_rating(page, rating)
        except ValueError:
            pass

        # Try to find and click the option
        strategies = [
            # Radio/checkbox by label text
            f"label:has-text('{target}')",
            f"input[value='{target}' i]",
            f"label:text-matches('{target}', 'i')",
            # Rating stars or numbered options
            f"[data-value='{target}']",
            f"[aria-label*='{target}' i]",
            # General text match
            f"text='{target}'",
        ]

        for selector in strategies:
            try:
                element = page.locator(selector).first
                if await element.is_visible(timeout=1000):
                    await element.click()
                    logger.info(f"Selected: {target}")
                    return True
            except Exception:
                continue

        logger.warning(f"Could not find option: {target}")
        return False

    async def _select_rating(self, page, rating: int) -> bool:
        """Select a numeric rating."""

        # Strategy 1: Look for radio buttons with value
        try:
            radio = page.locator(f"input[type='radio'][value='{rating}']").first
            if await radio.is_visible(timeout=1000):
                await radio.click()
                logger.info(f"Selected rating: {rating}")
                return True
        except Exception:
            pass

        # Strategy 2: Look for label containing the number
        try:
            label = page.locator(f"label:text-matches('^{rating}$')").first
            if await label.is_visible(timeout=1000):
                await label.click()
                logger.info(f"Selected rating via label: {rating}")
                return True
        except Exception:
            pass

        # Strategy 3: Find all radio buttons and click the one at position (rating-1)
        try:
            radios = page.locator("input[type='radio']")
            count = await radios.count()
            if count >= rating:
                # For 1-10 scale, click the (rating-1)th element (0-indexed)
                # But we need to handle 1-5, 1-10, 0-10 scales
                # Try to find the right one by checking values
                for i in range(count):
                    radio = radios.nth(i)
                    value = await radio.get_attribute("value")
                    if value and str(rating) == value:
                        await radio.click()
                        logger.info(f"Selected rating by value: {rating}")
                        return True
        except Exception:
            pass

        # Strategy 4: Click nth radio button (rating - 1 for 1-indexed)
        try:
            radios = page.locator("input[type='radio']:visible")
            count = await radios.count()
            if count > 0:
                # Assuming highest rating is last or rating matches position
                index = min(rating - 1, count - 1)
                await radios.nth(index).click()
                logger.info(f"Selected rating at position: {index}")
                return True
        except Exception:
            pass

        return False


async def run_vision_navigation(
        page,
        survey_url: str,
        mood: str = "happy",
        email: Optional[str] = None,
        survey_code: Optional[str] = None,
        max_steps: int = 50,
) -> dict:
    """
    Run vision-based survey navigation.

    Args:
        page: Playwright page object
        survey_url: Survey URL
        mood: User mood (happy, neutral, angry)
        email: Email for coupon delivery
        survey_code: Survey validation code
        max_steps: Maximum navigation steps

    Returns:
        Dict with success status, coupon code, etc.
    """
    from ..browser.interactor import PageInteractor

    # Pre-flight check: Verify vision backend is available
    from .ollama_host import detect_backend_async, detect_ollama_host_async
    backend = await detect_backend_async()

    if backend == "ollama":
        ollama_host = await detect_ollama_host_async()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health = await client.get(f"{ollama_host}/api/tags")
                if health.status_code != 200:
                    raise Exception(f"Ollama returned status {health.status_code}")
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {ollama_host}")
            return {
                "is_complete": False,
                "error": "Vision backend not available. Start Ollama or set DASHSCOPE_API_KEY.",
                "current_step": 0,
                "actions_taken": [],
            }
        except Exception as e:
            logger.warning(f"Ollama health check warning: {e}")
    else:
        logger.info("Using DashScope backend, skipping Ollama health check")

    navigator = VisionNavigator(mood=mood, email=email)
    interactor = PageInteractor()

    result: Dict[str, Any] = {
        "is_complete": False,
        "coupon_code": None,
        "error": None,
        "current_step": 0,
        "actions_taken": [],
    }

    try:
        # Navigate to survey URL
        logger.info(f"Navigating to: {survey_url}")
        await page.goto(survey_url, wait_until="networkidle")
        await asyncio.sleep(1)

        # Handle survey code entry if needed
        if survey_code:
            page_text = await page.inner_text("body")
            if any(phrase in page_text.lower() for phrase in ["survey code", "enter the", "digit code", "24-digit"]):
                logger.info("Detected code entry page, filling survey code")
                code_filled = await interactor.fill_survey_code(page, survey_code)

                if code_filled:
                    # Click Start button
                    start_btn = page.locator("button:has-text('Start'), input[value*='Start' i]").first
                    if await start_btn.is_visible(timeout=2000):
                        await start_btn.click()
                        await page.wait_for_load_state("networkidle")
                        await asyncio.sleep(1)

                        # Check for errors
                        error_text = await page.inner_text("body")
                        if any(err in error_text.lower() for err in ["already used", "invalid", "expired"]):
                            result["error"] = "Survey code already used or invalid"
                            return result

        # Main navigation loop
        last_page_url = ""
        same_page_count = 0

        for step in range(max_steps):
            result["current_step"] = step

            # Detect if we're stuck on the same page
            current_url = page.url
            if current_url == last_page_url:
                same_page_count += 1
            else:
                same_page_count = 0
                last_page_url = current_url

            # If stuck on same page for 3+ steps, try clicking Next
            if same_page_count >= 3:
                logger.warning(f"Stuck on same page for {same_page_count} steps, forcing Next click")
                clicked = await _try_click_next(page)
                if clicked:
                    same_page_count = 0
                    await asyncio.sleep(1)
                    continue

            # Analyze page with vision
            action = await navigator.analyze_page(page, step)
            result["actions_taken"].append(f"Step {step}: {action.action_type} - {action.target or action.value or ''}")

            # Check completion
            if action.action_type == "done":
                result["is_complete"] = True
                # Try to extract coupon code
                result["coupon_code"] = await _extract_coupon_from_page(page)
                logger.info(f"Survey complete! Coupon: {result['coupon_code']}")
                break

            # Check errors
            if action.action_type == "error":
                result["error"] = action.target or action.reasoning
                logger.error(f"Survey error: {result['error']}")
                break

            # Execute action
            success = await navigator.execute_action(page, action)

            if not success and action.action_type not in ("wait",):
                logger.warning(f"Action failed, retrying...")

            # After SELECT action, automatically try to click Next/Submit
            if action.action_type == "select" and success:
                await asyncio.sleep(0.3)
                # Check if there's a Next button to click
                next_clicked = await _try_click_next(page)
                if next_clicked:
                    logger.info("Auto-clicked Next after selection")
                    await asyncio.sleep(0.5)

            # Small delay between actions
            await asyncio.sleep(0.5)

        else:
            result["error"] = "Maximum steps reached"
            logger.warning("Max steps reached without completion")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Vision navigation failed: {e}")

    return result


async def _try_click_next(page) -> bool:
    """
    Try to find and click a Next/Submit/Continue button.

    Returns True if a button was clicked.
    """
    # Common button selectors for survey navigation
    next_selectors = [
        "button:has-text('Next')",
        "input[value='Next' i]",
        "button:has-text('Continue')",
        "input[value='Continue' i]",
        "button:has-text('Submit')",
        "input[type='submit']",
        "button:has-text('>')",
        "a:has-text('Next')",
        ".next-button",
        "#NextButton",
        "[class*='next']",
    ]

    for selector in next_selectors:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=500):
                # Check if it's enabled
                is_disabled = await btn.get_attribute("disabled")
                if not is_disabled:
                    await btn.click()
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    return True
        except Exception:
            continue

    return False


async def _extract_coupon_from_page(page) -> Optional[str]:
    """Try to extract coupon code from completion page."""
    import re

    try:
        page_text = await page.inner_text("body")

        # Common coupon patterns
        patterns = [
            r'\b[A-Z]{2,4}[-\s]?\d{4,8}\b',
            r'\b\d{4,8}[-\s]?[A-Z]{2,4}\b',
            r'\b[A-Z]{2,4}\d{2,4}[A-Z]{2,4}\b',
        ]

        blacklist = ["INTEREST", "COMPLETE", "SURVEY", "EXPRESS", "FEEDBACK"]

        for pattern in patterns:
            matches = re.findall(pattern, page_text)
            for match in matches:
                if match.upper() not in blacklist and any(c.isdigit() for c in match):
                    return match

        return None

    except Exception:
        return None