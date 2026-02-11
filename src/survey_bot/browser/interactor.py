from __future__ import annotations

import asyncio
import logging
import random
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union, Literal

from ..models.page_state import (
    ElementType,
    QuestionType,
    WebElement,
)


if TYPE_CHECKING:
    from playwright.async_api import Page, Locator


__all__ = ["PageInteractor"]

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_SCREENSHOT_DIR = Path("screenshots")
DEFAULT_CLICK_TIMEOUT = 5000  # 5 seconds
DEFAULT_TYPE_DELAY = 50  # ms between keystrokes
DEFAULT_ACTION_DELAY = (100, 300)  # Random delay range between actions


class PageInteractor:
    """
    Handles all interactions with web page elements.
    
    Provides methods for clicking, typing, selecting, and other
    interactions with proper error handling, retries, and
    human-like delays to avoid bot detection.
    
    Attributes:
        click_timeout: Timeout for click operations in ms.
        type_delay: Delay between keystrokes in ms.
        action_delay: Range for random delays between actions.
        screenshot_dir: Directory for saving screenshots.
        max_retries: Maximum retry attempts for failed actions.
    """
    
    def __init__(
        self,
        click_timeout: int = DEFAULT_CLICK_TIMEOUT,
        type_delay: int = DEFAULT_TYPE_DELAY,
        action_delay: tuple[int, int] = DEFAULT_ACTION_DELAY,
        screenshot_dir: Union[Path, str] = DEFAULT_SCREENSHOT_DIR,
        max_retries: int = 3,
        human_like: bool = True,
    ) -> None:
        """
        Initialize the PageInteractor.
        
        Args:
            click_timeout: Timeout for click operations in milliseconds.
            type_delay: Base delay between keystrokes in milliseconds.
            action_delay: (min, max) range for random delays between actions.
            screenshot_dir: Directory for saving screenshots.
            max_retries: Maximum number of retry attempts.
            human_like: Whether to add human-like variations to actions.
        """
        self.click_timeout = click_timeout
        self.type_delay = type_delay
        self.action_delay = action_delay
        self.screenshot_dir = Path(screenshot_dir)
        self.max_retries = max_retries
        self.human_like = human_like
        
        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(
            f"PageInteractor initialized: timeout={click_timeout}ms, "
            f"type_delay={type_delay}ms, human_like={human_like}"
        )
    
    async def _random_delay(self, min_ms: Optional[int] = None, max_ms: Optional[int] = None) -> None:
        """Add a random delay for human-like behavior."""
        if not self.human_like:
            return
        
        min_delay = min_ms or self.action_delay[0]
        max_delay = max_ms or self.action_delay[1]
        delay = random.randint(min_delay, max_delay)
        await asyncio.sleep(delay / 1000)
    
    async def _get_locator(self, page: Page, element: WebElement) -> Locator:
        """
        Get a Playwright Locator for the element.
        
        Tries multiple strategies to locate the element.
        """
        # Try by ID first (most reliable)
        if element.element_id and not element.element_id.startswith("auto_"):
            locator = page.locator(f"#{element.element_id}")
            if await locator.count() > 0:
                return locator.first
        
        # Try by selector
        if element.selector and element.selector != "unknown":
            try:
                locator = page.locator(element.selector)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                pass
        
        # Try by text content
        if element.text_content:
            # Escape special characters for text matching
            safe_text = element.text_content.replace('"', '\\"')
            
            if element.element_type == ElementType.BUTTON:
                locator = page.locator(f"button:has-text(\"{safe_text}\")")
                if await locator.count() > 0:
                    return locator.first
            
            if element.element_type == ElementType.LINK:
                locator = page.locator(f"a:has-text(\"{safe_text}\")")
                if await locator.count() > 0:
                    return locator.first
        
        # Try by aria-label
        if element.aria_label:
            locator = page.locator(f"[aria-label=\"{element.aria_label}\"]")
            if await locator.count() > 0:
                return locator.first
        
        # Fall back to generic selector with text
        if element.text_content:
            locator = page.get_by_text(element.text_content, exact=False)
            if await locator.count() > 0:
                return locator.first
        
        raise ValueError(f"Could not locate element: {element.element_id}")
    
    async def click_element(
        self,
        page: Page,
        element: WebElement,
        *,
        force: bool = False,
        wait_for_navigation: bool = False,
    ) -> bool:
        """
        Click an element with retry logic and human-like behavior.
        
        Args:
            page: Playwright Page object.
            element: WebElement to click.
            force: Force click even if element is not visible.
            wait_for_navigation: Wait for page navigation after click.
            
        Returns:
            True if click succeeded, False otherwise.
        """
        logger.info(f"Clicking element: {element.get_display_text()[:50]}")
        
        for attempt in range(self.max_retries):
            try:
                # Get locator
                locator = await self._get_locator(page, element)
                
                # Scroll into view
                await locator.scroll_into_view_if_needed()
                await self._random_delay(50, 150)
                
                # Wait for element to be clickable
                await locator.wait_for(state="visible", timeout=self.click_timeout)
                
                # Add human-like delay before clicking
                await self._random_delay()
                
                # Perform click
                if wait_for_navigation:
                    async with page.expect_navigation(timeout=10000):
                        await locator.click(force=force, timeout=self.click_timeout)
                else:
                    await locator.click(force=force, timeout=self.click_timeout)
                
                logger.debug(f"Click successful on attempt {attempt + 1}")
                
                # Small delay after click
                await self._random_delay(100, 300)
                
                return True
                
            except Exception as e:
                logger.warning(f"Click attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(0.5)
                else:
                    logger.error(f"Click failed after {self.max_retries} attempts")
                    return False
        
        return False
    
    async def fill_survey_code(
        self,
        page: Page,
        survey_code: str,
    ) -> bool:
        """
        Fill survey code into multiple input fields.
        
        Handles the common pattern where survey codes are split across
        multiple input fields (e.g., 6 fields for a 24-digit code).
        
        Args:
            page: Playwright Page object.
            survey_code: Full survey code (with or without dashes).
            
        Returns:
            True if code was filled successfully, False otherwise.
        """
        # Clean the code - remove dashes and spaces
        clean_code = survey_code.replace("-", "").replace(" ", "")
        logger.info(f"Filling survey code: {clean_code[:4]}...{clean_code[-4:]} ({len(clean_code)} digits)")
        
        try:
            # Strategy 1: Look for multiple sequential text inputs
            # Common patterns: 6 fields of 4 digits, 4 fields of 6 digits, etc.
            text_inputs = page.locator("input[type='text'], input:not([type])")
            count = await text_inputs.count()
            
            if count >= 4:
                # Filter to only visible, empty-ish inputs that look like code fields
                code_inputs = []
                for i in range(count):
                    inp = text_inputs.nth(i)
                    try:
                        if await inp.is_visible():
                            # Check if it looks like a code input (short maxlength, numeric pattern)
                            maxlen = await inp.get_attribute("maxlength")
                            input_type = await inp.get_attribute("type")
                            name = await inp.get_attribute("name") or ""
                            
                            # Likely a code field if maxlength is 4-6 or name contains code/digit
                            if maxlen and int(maxlen) <= 6:
                                code_inputs.append(inp)
                            elif "code" in name.lower() or "digit" in name.lower():
                                code_inputs.append(inp)
                    except Exception:
                        continue
                
                if len(code_inputs) >= 4:
                    logger.info(f"Found {len(code_inputs)} code input fields")
                    
                    # Calculate segment size
                    segment_size = len(clean_code) // len(code_inputs)
                    
                    for i, inp in enumerate(code_inputs):
                        start = i * segment_size
                        end = start + segment_size
                        segment = clean_code[start:end]
                        
                        if segment:
                            await inp.click()
                            await self._random_delay(50, 100)
                            await inp.fill(segment)
                            await self._random_delay(100, 200)
                            logger.debug(f"Filled field {i+1}: {segment}")
                    
                    logger.info("Survey code filled successfully")
                    return True
            
            # Strategy 2: Single input field for full code
            single_input = page.locator("input[name*='code' i], input[id*='code' i], input[placeholder*='code' i]")
            if await single_input.count() > 0:
                inp = single_input.first
                if await inp.is_visible():
                    await inp.click()
                    await inp.fill(survey_code)  # Use original with dashes
                    logger.info("Survey code filled in single field")
                    return True
            
            logger.warning("Could not find survey code input fields")
            return False
            
        except Exception as e:
            logger.error(f"Error filling survey code: {e}")
            return False
    
    async def fill_text(
        self,
        page: Page,
        element: WebElement,
        text: str,
        *,
        clear_first: bool = True,
        human_like_typing: bool = True,
    ) -> bool:
        """
        Fill a text input or textarea with human-like typing.
        
        Args:
            page: Playwright Page object.
            element: WebElement (text input or textarea).
            text: Text to enter.
            clear_first: Whether to clear existing text first.
            human_like_typing: Whether to type character by character.
            
        Returns:
            True if fill succeeded, False otherwise.
        """
        logger.info(f"Filling text in: {element.get_display_text()[:30]} ({len(text)} chars)")
        
        for attempt in range(self.max_retries):
            try:
                # Get locator
                locator = await self._get_locator(page, element)
                
                # Scroll into view
                await locator.scroll_into_view_if_needed()
                await self._random_delay(50, 100)
                
                # Click to focus
                await locator.click()
                await self._random_delay(50, 150)
                
                # Clear existing text if requested
                if clear_first:
                    await locator.fill("")
                    await self._random_delay(50, 100)
                
                # Type text
                if human_like_typing and self.human_like:
                    # Type character by character with variable delays
                    await self._human_type(locator, text)
                else:
                    # Fill instantly
                    await locator.fill(text)
                
                logger.debug(f"Text fill successful on attempt {attempt + 1}")
                
                # Small delay after typing
                await self._random_delay(100, 200)
                
                return True
                
            except Exception as e:
                logger.warning(f"Fill attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    logger.error(f"Fill failed after {self.max_retries} attempts")
                    return False
        
        return False
    
    async def _human_type(self, locator: Locator, text: str) -> None:
        """Type text character by character with human-like delays."""
        for char in text:
            await locator.press_sequentially(char, delay=self._get_typing_delay())
    
    def _get_typing_delay(self) -> int:
        """Get a random typing delay for human-like behavior."""
        if not self.human_like:
            return self.type_delay
        
        # Add variation: faster for common letters, slower for punctuation
        base = self.type_delay
        variation = random.randint(-20, 40)
        return max(10, base + variation)
    
    async def select_rating(
        self,
        page: Page,
        rating: int,
        max_rating: int = 10,
    ) -> bool:
        """
        Select a rating value from various rating UI patterns.
        
        Handles:
        - Radio button scales (1-10, 1-5)
        - Star ratings
        - NPS scales
        - Slider-like inputs
        
        Args:
            page: Playwright Page object.
            rating: Desired rating value.
            max_rating: Maximum rating on the scale (for scaling).
            
        Returns:
            True if rating selection succeeded, False otherwise.
        """
        logger.info(f"Selecting rating: {rating}/{max_rating}")
        
        # Strategy 1: Find radio buttons with rating values
        success = await self._select_rating_radio(page, rating, max_rating)
        if success:
            return True
        
        # Strategy 2: Find star rating elements
        success = await self._select_rating_stars(page, rating, max_rating)
        if success:
            return True
        
        # Strategy 3: Find buttons with rating numbers
        success = await self._select_rating_buttons(page, rating, max_rating)
        if success:
            return True
        
        # Strategy 4: Look for elements with data attributes
        success = await self._select_rating_data_attr(page, rating, max_rating)
        if success:
            return True
        
        logger.warning(f"Could not find rating element for value {rating}")
        return False
    
    async def _select_rating_radio(
        self,
        page: Page,
        rating: int,
        max_rating: int,
    ) -> bool:
        """Select rating via radio buttons."""
        try:
            # Find radio buttons
            radios = page.locator("input[type='radio']")
            count = await radios.count()
            
            if count == 0:
                return False
            
            # Try to find radio with matching value
            for i in range(count):
                radio = radios.nth(i)
                value = await radio.get_attribute("value")
                
                if value and value.isdigit() and int(value) == rating:
                    await self._random_delay()
                    await radio.click(force=True)
                    logger.debug(f"Selected radio with value={rating}")
                    return True
            
            # If no exact value match, try by position (for 1-N scales)
            if count >= 3:
                # Scale the rating to the number of options
                if max_rating == 10 and count == 5:
                    # Convert 10-point to 5-point scale
                    target_index = max(0, min(count - 1, (rating - 1) // 2))
                elif max_rating == 10 and count == 10:
                    target_index = rating - 1
                elif max_rating == 5 and count == 5:
                    target_index = rating - 1
                else:
                    # Generic scaling
                    target_index = int((rating / max_rating) * (count - 1))
                
                target_index = max(0, min(count - 1, target_index))
                
                radio = radios.nth(target_index)
                await self._random_delay()
                await radio.click(force=True)
                logger.debug(f"Selected radio at index {target_index}")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Radio rating selection failed: {e}")
            return False
    
    async def _select_rating_stars(
        self,
        page: Page,
        rating: int,
        max_rating: int,
    ) -> bool:
        """Select rating via star elements."""
        try:
            # Common star rating selectors
            star_selectors = [
                "[class*='star']",
                "[class*='rating'] span",
                "[class*='rating'] i",
                ".stars > *",
                "[data-rating]",
            ]
            
            for selector in star_selectors:
                stars = page.locator(selector)
                count = await stars.count()
                
                if count >= 3:
                    # Scale rating to star count
                    target_index = int((rating / max_rating) * count) - 1
                    target_index = max(0, min(count - 1, target_index))
                    
                    star = stars.nth(target_index)
                    if await star.is_visible():
                        await self._random_delay()
                        await star.click()
                        logger.debug(f"Clicked star at index {target_index}")
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Star rating selection failed: {e}")
            return False
    
    async def _select_rating_buttons(
        self,
        page: Page,
        rating: int,
        max_rating: int,
    ) -> bool:
        """Select rating via numbered buttons."""
        try:
            # Look for buttons or links with the rating number
            selectors = [
                f"button:has-text('{rating}')",
                f"[role='button']:has-text('{rating}')",
                f"a:has-text('{rating}')",
            ]
            
            for selector in selectors:
                element = page.locator(selector).first
                if await element.count() > 0 and await element.is_visible():
                    await self._random_delay()
                    await element.click()
                    logger.debug(f"Clicked rating button: {rating}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Button rating selection failed: {e}")
            return False
    
    async def _select_rating_data_attr(
        self,
        page: Page,
        rating: int,
        max_rating: int,
    ) -> bool:
        """Select rating via data attributes."""
        try:
            # Look for elements with data-value or data-rating
            selectors = [
                f"[data-value='{rating}']",
                f"[data-rating='{rating}']",
                f"[data-score='{rating}']",
            ]
            
            for selector in selectors:
                element = page.locator(selector).first
                if await element.count() > 0 and await element.is_visible():
                    await self._random_delay()
                    await element.click()
                    logger.debug(f"Clicked element with data attr: {rating}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Data attribute rating selection failed: {e}")
            return False
    
    async def select_option(
        self,
        page: Page,
        option_text: str,
        *,
        fuzzy_threshold: float = 0.6,
    ) -> bool:
        """
        Select an option from multiple choice questions.
        
        Uses fuzzy matching to handle slight text differences between
        the expected option and what's actually on the page.
        
        Args:
            page: Playwright Page object.
            option_text: Text of the option to select.
            fuzzy_threshold: Minimum similarity score (0-1) for fuzzy matching.
            
        Returns:
            True if option was selected, False otherwise.
        """
        logger.info(f"Selecting option: '{option_text[:50]}'")
        
        # Strategy 1: Exact text match on radio buttons
        success = await self._select_option_exact(page, option_text)
        if success:
            return True
        
        # Strategy 2: Fuzzy text match
        success = await self._select_option_fuzzy(page, option_text, fuzzy_threshold)
        if success:
            return True
        
        # Strategy 3: Try label associations
        success = await self._select_option_by_label(page, option_text, fuzzy_threshold)
        if success:
            return True
        
        logger.warning(f"Could not find option matching: '{option_text[:30]}'")
        return False
    
    async def _select_option_exact(self, page: Page, option_text: str) -> bool:
        """Select option by exact text match."""
        try:
            # Try radio buttons first
            radio_selectors = [
                f"input[type='radio'][value='{option_text}']",
                f"label:has-text('{option_text}') input[type='radio']",
            ]
            
            for selector in radio_selectors:
                element = page.locator(selector).first
                if await element.count() > 0:
                    await self._random_delay()
                    await element.click(force=True)
                    logger.debug(f"Selected radio with exact match: {option_text[:30]}")
                    return True
            
            # Try clicking label or container with text
            label = page.locator(f"label:has-text('{option_text}')").first
            if await label.count() > 0 and await label.is_visible():
                await self._random_delay()
                await label.click()
                logger.debug(f"Clicked label: {option_text[:30]}")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Exact option selection failed: {e}")
            return False
    
    async def _select_option_fuzzy(
        self,
        page: Page,
        option_text: str,
        threshold: float,
    ) -> bool:
        """Select option using fuzzy text matching."""
        try:
            # Get all radio buttons and their labels
            radios = page.locator("input[type='radio']")
            count = await radios.count()
            
            best_match: Optional[Tuple[int, float]] = None
            
            for i in range(count):
                radio = radios.nth(i)
                
                # Get associated label text
                radio_id = await radio.get_attribute("id")
                label_text = ""
                
                if radio_id:
                    label = page.locator(f"label[for='{radio_id}']")
                    if await label.count() > 0:
                        label_text = await label.inner_text()
                
                # Also check parent label
                if not label_text:
                    parent = radio.locator("xpath=..")
                    try:
                        label_text = await parent.inner_text()
                    except Exception:
                        pass
                
                # Check value attribute
                if not label_text:
                    label_text = await radio.get_attribute("value") or ""
                
                # Calculate similarity
                similarity = self._text_similarity(option_text, label_text)
                
                if similarity >= threshold:
                    if best_match is None or similarity > best_match[1]:
                        best_match = (i, similarity)
            
            if best_match:
                radio = radios.nth(best_match[0])
                await self._random_delay()
                await radio.click(force=True)
                logger.debug(f"Selected fuzzy match (similarity: {best_match[1]:.2f})")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Fuzzy option selection failed: {e}")
            return False
    
    async def _select_option_by_label(
        self,
        page: Page,
        option_text: str,
        threshold: float,
    ) -> bool:
        """Select option by finding and clicking the label."""
        try:
            # Get all labels
            labels = page.locator("label")
            count = await labels.count()
            
            best_match: Optional[Tuple[Locator, float]] = None
            
            for i in range(count):
                label = labels.nth(i)
                label_text = await label.inner_text()
                
                similarity = self._text_similarity(option_text, label_text)
                
                if similarity >= threshold:
                    if best_match is None or similarity > best_match[1]:
                        best_match = (label, similarity)
            
            if best_match:
                await self._random_delay()
                await best_match[0].click()
                logger.debug(f"Clicked label (similarity: {best_match[1]:.2f})")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Label option selection failed: {e}")
            return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings (0-1)."""
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, t1, t2).ratio()
    
    async def select_dropdown(
        self,
        page: Page,
        element: WebElement,
        option_text: str,
    ) -> bool:
        """
        Select an option from a dropdown/select element.
        
        Args:
            page: Playwright Page object.
            element: The select element.
            option_text: Text of the option to select.
            
        Returns:
            True if selection succeeded, False otherwise.
        """
        logger.info(f"Selecting dropdown option: '{option_text[:30]}'")
        
        try:
            locator = await self._get_locator(page, element)
            
            await self._random_delay()
            
            # Try selecting by label (visible text)
            await locator.select_option(label=option_text)
            
            logger.debug(f"Dropdown selection successful: {option_text[:30]}")
            return True
            
        except Exception as e:
            logger.warning(f"Dropdown selection by label failed: {e}")
            
            # Try fuzzy match on options
            try:
                options = page.locator(f"{element.selector} option")
                count = await options.count()
                
                for i in range(count):
                    opt = options.nth(i)
                    opt_text = await opt.inner_text()
                    
                    if self._text_similarity(option_text, opt_text) >= 0.6:
                        value = await opt.get_attribute("value")
                        locator = await self._get_locator(page, element)
                        await locator.select_option(value=value)
                        logger.debug(f"Dropdown fuzzy selection: {opt_text[:30]}")
                        return True
                        
            except Exception as e2:
                logger.error(f"Dropdown selection failed: {e2}")
            
            return False
    
    async def check_checkbox(
        self,
        page: Page,
        element: WebElement,
        check: bool = True,
    ) -> bool:
        """
        Check or uncheck a checkbox.
        
        Args:
            page: Playwright Page object.
            element: The checkbox element.
            check: True to check, False to uncheck.
            
        Returns:
            True if action succeeded, False otherwise.
        """
        logger.info(f"{'Checking' if check else 'Unchecking'} checkbox: {element.get_display_text()[:30]}")
        
        try:
            locator = await self._get_locator(page, element)
            
            current_state = await locator.is_checked()
            
            if current_state != check:
                await self._random_delay()
                await locator.click(force=True)
                logger.debug(f"Checkbox toggled to: {check}")
            else:
                logger.debug(f"Checkbox already in desired state: {check}")
            
            return True
            
        except Exception as e:
            logger.error(f"Checkbox action failed: {e}")
            return False
    
    async def take_screenshot(
        self,
        page: Page,
        name: str = "screenshot",
        *,
        full_page: bool = True,
    ) -> str:
        """
        Take a screenshot of the current page.
        
        Args:
            page: Playwright Page object.
            name: Base name for the screenshot file.
            full_page: Whether to capture the full scrollable page.
            
        Returns:
            Path to the saved screenshot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        
        try:
            await page.screenshot(path=str(filepath), full_page=full_page)
            logger.info(f"Screenshot saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            raise
    
    async def wait_for_element(
        self,
        page: Page,
        selector: str,
        timeout: Optional[int] = None,
        state: Literal["attached", "detached", "hidden", "visible"] = "visible",
    ) -> bool:
        """
        Wait for an element to appear on the page.
        
        Args:
            page: Playwright Page object.
            selector: CSS selector for the element.
            timeout: Timeout in milliseconds.
            state: State to wait for ('visible', 'hidden', 'attached').
            
        Returns:
            True if element appeared, False if timeout.
        """
        try:
            await page.locator(selector).wait_for(
                state=state,
                timeout=timeout or self.click_timeout
            )
            return True
        except Exception:
            return False
    
    async def press_key(
        self,
        page: Page,
        key: str,
    ) -> bool:
        """
        Press a keyboard key.
        
        Args:
            page: Playwright Page object.
            key: Key to press (e.g., 'Enter', 'Tab', 'Escape').
            
        Returns:
            True if successful.
        """
        try:
            await self._random_delay(50, 100)
            await page.keyboard.press(key)
            logger.debug(f"Pressed key: {key}")
            return True
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False
