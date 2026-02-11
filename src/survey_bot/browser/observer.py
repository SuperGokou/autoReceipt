"""
Page Observer for extracting survey page state.

This module provides the PageObserver class which analyzes web pages
to extract interactive elements, detect question types, and identify
survey completion states.

The observer is designed specifically for survey automation and can:
- Extract all clickable/interactive elements
- Classify question types (rating, text, multiple choice, etc.)
- Detect coupon/validation codes on completion pages
- Find navigation buttons (Next, Submit, Continue)

Example Usage:
    >>> from playwright.async_api import Page
    >>> from survey_bot.browser.observer import PageObserver
    >>> 
    >>> observer = PageObserver()
    >>> state = await observer.get_page_state(page)
    >>> print(f"Question type: {state.question_type}")
    >>> print(f"Found {len(state.elements)} interactive elements")
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING, Optional, Tuple, List

from ..models.page_state import (
    BoundingBox,
    ElementType,
    PageState,
    QuestionType,
    WebElement,
)


if TYPE_CHECKING:
    from playwright.async_api import Page, Locator


__all__ = ["PageObserver"]

logger = logging.getLogger(__name__)


# Selectors for finding interactive elements
INTERACTIVE_SELECTORS = {
    ElementType.BUTTON: [
        "button",
        "input[type='submit']",
        "input[type='button']",
        "[role='button']",
        ".btn",
        ".button",
    ],
    ElementType.LINK: [
        "a[href]",
    ],
    ElementType.RADIO: [
        "input[type='radio']",
        "[role='radio']",
    ],
    ElementType.CHECKBOX: [
        "input[type='checkbox']",
        "[role='checkbox']",
    ],
    ElementType.TEXT_INPUT: [
        "input[type='text']",
        "input[type='email']",
        "input[type='tel']",
        "input[type='number']",
        "input:not([type])",  # Default input type is text
    ],
    ElementType.TEXTAREA: [
        "textarea",
    ],
    ElementType.SELECT: [
        "select",
    ],
    ElementType.STAR_RATING: [
        "[class*='star']",
        "[class*='rating']",
        "[data-rating]",
        ".stars",
    ],
}

# Keywords for detecting submit/next buttons
SUBMIT_KEYWORDS = [
    "next", "continue", "submit", "proceed", "finish",
    "done", "complete", "send", "go", "start",
    "siguiente", "continuar", "enviar",  # Spanish
    "weiter", "absenden",  # German
    "suivant", "envoyer",  # French
]

# Keywords for detecting coupon/validation codes
COUPON_KEYWORDS = [
    "validation code", "coupon code", "voucher code", "discount code",
    "confirmation code", "reward code", "your code",
    "here is your code", "thank you for completing",
    "código", "cupón",  # Spanish
]

# Error keywords that indicate survey failed
ERROR_KEYWORDS = [
    "already been used", "already used", "already completed",
    "invalid code", "code is invalid", "code not found",
    "expired", "no longer valid", "not valid",
    "error", "sorry", "unable to process",
    "code has been redeemed", "previously used",
]

# Patterns for extracting coupon codes (more specific to avoid false positives)
COUPON_PATTERNS = [
    r'\b[A-Z]{2,4}[-\s]?\d{4,8}\b',      # ABC-12345 or ABC 12345
    r'\b\d{4,8}[-\s]?[A-Z]{2,4}\b',      # 12345-ABC
    r'\b[A-Z]{2,4}\d{2,4}[A-Z]{2,4}\b',  # ABC12DEF (mixed)
    r'\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b',  # 123-456-789
]

# Words that match patterns but are NOT coupons
COUPON_BLACKLIST = [
    "INTEREST", "COMPLETE", "COMPLETED", "SUBMIT", "SURVEY", "FEEDBACK",
    "EXPRESS", "WELCOME", "PLEASE", "THANKS", "THANK", "CONGRATULATIONS",
    "RESTAURANT", "CUSTOMER", "SERVICE", "QUALITY", "EXPERIENCE",
    "SATISFIED", "LIKELY", "UNLIKELY", "EXCELLENT", "AVERAGE", "VISIT",
    "PURCHASE", "REWARD", "REWARDS", "COUPON", "DISCOUNT", "VALIDATION",
    "RECEIPT", "CONTINUE", "NEXT", "BACK", "START", "BEGIN", "FINISH",
]


class PageObserver:
    """
    Observes and extracts state from survey web pages.
    
    The PageObserver is responsible for analyzing page content
    and structure to provide the Navigator with actionable information
    about the current survey state.
    
    Attributes:
        include_hidden: Whether to include hidden elements.
        max_elements: Maximum number of elements to extract.
        text_truncate_length: Max length for text content.
    """
    
    def __init__(
        self,
        include_hidden: bool = False,
        max_elements: int = 100,
        text_truncate_length: int = 500,
    ) -> None:
        """
        Initialize the PageObserver.
        
        Args:
            include_hidden: Include hidden elements in extraction.
            max_elements: Max elements to extract (prevents memory issues).
            text_truncate_length: Max length for extracted text.
        """
        self.include_hidden = include_hidden
        self.max_elements = max_elements
        self.text_truncate_length = text_truncate_length
        
        logger.debug("PageObserver initialized")
    
    async def get_page_state(self, page: Page) -> PageState:
        """
        Extract complete state from the current page.
        
        Combines element extraction, question type detection,
        and coupon detection into a single PageState object.
        
        Args:
            page: Playwright Page object.
            
        Returns:
            PageState with all extracted information.
        """
        logger.debug(f"Extracting page state from: {page.url}")
        
        # Extract basic page info
        url = page.url
        title = await page.title()
        
        # Get page text content (truncated)
        page_text = await self._get_page_text(page)
        
        # Extract interactive elements
        elements = await self.extract_interactive_elements(page)
        
        # Detect question type
        question_type = await self.detect_question_type(page, elements)
        
        # Try to extract question text
        question_text = await self._extract_question_text(page)
        
        # Check for coupon/completion
        has_coupon, coupon_code = await self._detect_coupon(page, page_text)
        
        # Check for error messages
        error_message = await self._detect_error_message(page)
        
        # Build state object
        state = PageState(
            url=url,
            title=title,
            elements=elements,
            question_type=question_type,
            question_text=question_text,
            has_coupon=has_coupon,
            coupon_code=coupon_code,
            page_text=page_text[:self.text_truncate_length],
            error_message=error_message,
        )
        
        logger.info(
            f"Page state: type={question_type.value}, "
            f"elements={len(elements)}, has_coupon={has_coupon}"
        )
        
        # Debug: If 0 elements found, this likely indicates bot detection or loading issue
        if len(elements) == 0:
            logger.warning(f"No elements found! URL: {url}")
            logger.warning(f"Page title: {title}")
            logger.warning(f"Page text length: {len(page_text)} chars")
            # Check for common bot detection indicators
            page_lower = page_text.lower()
            if "cloudflare" in page_lower:
                logger.error("DETECTED: Cloudflare protection - headless browser blocked")
            if "verify you are human" in page_lower:
                logger.error("DETECTED: Bot verification required")
            if "access denied" in page_lower:
                logger.error("DETECTED: Access denied page")
            if "enable javascript" in page_lower:
                logger.error("DETECTED: JavaScript required but not executing")
        
        return state
    
    async def extract_interactive_elements(
        self,
        page: Page,
    ) -> list[WebElement]:
        """
        Extract all interactive elements from the page.
        
        Finds buttons, inputs, radio buttons, checkboxes, links,
        and other interactive elements.
        
        Args:
            page: Playwright Page object.
            
        Returns:
            List of WebElement objects.
        """
        elements: list[WebElement] = []
        seen_ids: set[str] = set()
        
        for element_type, selectors in INTERACTIVE_SELECTORS.items():
            for selector in selectors:
                try:
                    locators = page.locator(selector)
                    count = await locators.count()
                    
                    for i in range(min(count, self.max_elements - len(elements))):
                        if len(elements) >= self.max_elements:
                            break
                        
                        locator = locators.nth(i)
                        element = await self._extract_element(
                            locator, element_type, seen_ids
                        )
                        
                        if element:
                            elements.append(element)
                            seen_ids.add(element.element_id)
                            
                except Exception as e:
                    logger.debug(f"Error extracting {selector}: {e}")
        
        logger.debug(f"Extracted {len(elements)} interactive elements")
        return elements
    
    async def _extract_element(
        self,
        locator: Locator,
        element_type: ElementType,
        seen_ids: set[str],
    ) -> Optional[WebElement]:
        """
        Extract data from a single element.
        
        Args:
            locator: Playwright Locator for the element.
            element_type: Pre-determined element type.
            seen_ids: Set of already-seen IDs to avoid duplicates.
            
        Returns:
            WebElement or None if extraction fails.
        """
        try:
            # Check visibility
            is_visible = await locator.is_visible()
            if not is_visible and not self.include_hidden:
                return None
            
            # Get element handle for attribute extraction
            handle = await locator.element_handle()
            if not handle:
                return None
            
            # Extract basic attributes
            element_id = await handle.get_attribute("id") or ""
            name = await handle.get_attribute("name") or ""
            
            # Generate unique ID if none exists
            if not element_id:
                element_id = f"auto_{element_type.value}_{uuid.uuid4().hex[:8]}"
            
            # Skip if already seen
            if element_id in seen_ids:
                return None
            
            # Extract text content
            text_content = await locator.inner_text() if await locator.count() > 0 else ""
            text_content = text_content.strip()[:200]  # Truncate
            
            # Extract other attributes
            aria_label = await handle.get_attribute("aria-label")
            placeholder = await handle.get_attribute("placeholder")
            
            # For text inputs and textareas, use input_value() to get typed content
            # get_attribute("value") only returns the initial HTML value attribute
            value = None
            if element_type in (ElementType.TEXT_INPUT, ElementType.TEXTAREA):
                try:
                    value = await locator.input_value()
                except Exception:
                    value = await handle.get_attribute("value")
            else:
                value = await handle.get_attribute("value")
            
            is_enabled = await locator.is_enabled()
            
            # Check if checked (for radio/checkbox)
            is_checked = None
            if element_type in (ElementType.RADIO, ElementType.CHECKBOX):
                is_checked = await locator.is_checked()
            
            # Get bounding box
            bbox_dict = await locator.bounding_box()
            bounding_box = None
            if bbox_dict:
                bounding_box = BoundingBox(
                    x=bbox_dict["x"],
                    y=bbox_dict["y"],
                    width=bbox_dict["width"],
                    height=bbox_dict["height"],
                )
            
            # Build selector
            selector = await self._build_selector(handle, element_id, name)
            
            # Collect other useful attributes
            attributes = {}
            for attr in ["class", "type", "href", "data-value", "data-rating"]:
                attr_value = await handle.get_attribute(attr)
                if attr_value:
                    attributes[attr] = attr_value
            
            return WebElement(
                element_id=element_id,
                element_type=element_type,
                text_content=text_content,
                aria_label=aria_label,
                placeholder=placeholder,
                is_visible=is_visible,
                is_enabled=is_enabled,
                is_checked=is_checked,
                bounding_box=bounding_box,
                attributes=attributes,
                selector=selector,
                value=value,
            )
            
        except Exception as e:
            logger.debug(f"Failed to extract element: {e}")
            return None
    
    async def _build_selector(
        self,
        handle,
        element_id: str,
        name: str,
    ) -> str:
        """Build a CSS selector for the element."""
        if element_id and not element_id.startswith("auto_"):
            return f"#{element_id}"
        if name:
            return f"[name='{name}']"
        
        # Fall back to a generic selector
        tag = await handle.evaluate("el => el.tagName.toLowerCase()")
        return tag or "unknown"
    
    async def detect_question_type(
        self,
        page: Page,
        elements: Optional[List[WebElement]] = None,
    ) -> QuestionType:
        """
        Detect the type of question on the current page.
        
        Analyzes page structure and elements to classify the
        type of survey question being asked.
        
        Args:
            page: Playwright Page object.
            elements: Pre-extracted elements (optional, will extract if None).
            
        Returns:
            Detected QuestionType.
        """
        if elements is None:
            elements = await self.extract_interactive_elements(page)
        
        # Count element types
        radios = [e for e in elements if e.element_type == ElementType.RADIO]
        checkboxes = [e for e in elements if e.element_type == ElementType.CHECKBOX]
        text_inputs = [e for e in elements if e.element_type in (
            ElementType.TEXT_INPUT, ElementType.TEXTAREA
        )]
        star_ratings = [e for e in elements if e.element_type == ElementType.STAR_RATING]
        buttons = [e for e in elements if e.element_type == ElementType.BUTTON]
        
        # Get page text for analysis
        page_text = await self._get_page_text(page)
        page_text_lower = page_text.lower()
        
        # Check for ERROR page first (code already used, invalid, etc.)
        for error_kw in ERROR_KEYWORDS:
            if error_kw in page_text_lower:
                logger.info(f"Detected ERROR page: '{error_kw}'")
                return QuestionType.UNKNOWN  # Return unknown so it doesn't claim success
        
        # Check for completion/coupon page (must have specific phrases + code)
        completion_phrases = ["thank you for completing", "your code is", "validation code", 
                             "coupon code", "here is your code", "your reward"]
        if any(phrase in page_text_lower for phrase in completion_phrases):
            # Verify by looking for code patterns that contain digits
            for pattern in COUPON_PATTERNS:
                matches = re.findall(pattern, page_text)
                for match in matches:
                    if any(c.isdigit() for c in match) and match.upper() not in COUPON_BLACKLIST:
                        logger.debug("Detected COMPLETION page (valid coupon found)")
                        return QuestionType.COMPLETION
        
        # Check for rating scale (stars, 1-10, NPS)
        if star_ratings:
            logger.debug("Detected RATING_SCALE (star elements found)")
            return QuestionType.RATING_SCALE
        
        # Check for numeric rating patterns
        rating_patterns = [
            r'\b(1|0)\s*[-–—]\s*(10|5)\b',  # "1-10" or "0-5"
            r'(rate|rating|score|scale)',
            r'(strongly disagree|disagree|neutral|agree|strongly agree)',
            r'(very unlikely|unlikely|neutral|likely|very likely)',
            r'(poor|fair|good|excellent)',
        ]
        for pattern in rating_patterns:
            if re.search(pattern, page_text_lower):
                # If we have radio buttons, it's likely a rating scale
                if len(radios) >= 3:
                    logger.debug("Detected RATING_SCALE (pattern + radios)")
                    return QuestionType.RATING_SCALE
        
        # Check for yes/no (exactly 2 radio buttons or specific text)
        if len(radios) == 2:
            radio_texts = " ".join(r.text_content.lower() for r in radios)
            if any(pair in radio_texts for pair in [
                "yes no", "no yes", "true false", "agree disagree"
            ]):
                logger.debug("Detected YES_NO (2 radios with yes/no)")
                return QuestionType.YES_NO
        
        # Check for multiple choice (3+ radio buttons)
        if len(radios) >= 3:
            logger.debug("Detected MULTIPLE_CHOICE (3+ radios)")
            return QuestionType.MULTIPLE_CHOICE
        
        # Check for multi-select (checkboxes)
        if len(checkboxes) >= 2:
            logger.debug("Detected MULTI_SELECT (checkboxes)")
            return QuestionType.MULTI_SELECT
        
        # Check for text input
        if text_inputs:
            # Filter out small inputs (might be for other purposes)
            large_inputs = [
                t for t in text_inputs 
                if t.element_type == ElementType.TEXTAREA or 
                (t.bounding_box and t.bounding_box.width > 200)
            ]
            if large_inputs:
                logger.debug("Detected TEXT_INPUT (textarea/large input)")
                return QuestionType.TEXT_INPUT
        
        # Check for dropdown
        selects = [e for e in elements if e.element_type == ElementType.SELECT]
        if selects:
            logger.debug("Detected DROPDOWN")
            return QuestionType.DROPDOWN
        
        # Check if it's just a navigation page (only has next button)
        submit_button = await self.find_submit_button(page, elements)
        if submit_button and len(radios) == 0 and len(checkboxes) == 0 and len(text_inputs) == 0:
            logger.debug("Detected NAVIGATION (only submit button)")
            return QuestionType.NAVIGATION
        
        logger.debug("Question type UNKNOWN")
        return QuestionType.UNKNOWN
    
    async def find_submit_button(
        self,
        page: Page,
        elements: Optional[List[WebElement]] = None,
    ) -> Optional[WebElement]:
        """
        Find the submit/next/continue button on the page.
        
        Uses semantic matching to find buttons that advance
        the survey to the next page.
        
        Args:
            page: Playwright Page object.
            elements: Pre-extracted elements (optional).
            
        Returns:
            WebElement for the submit button, or None if not found.
        """
        if elements is None:
            elements = await self.extract_interactive_elements(page)
        
        # Filter to buttons and links
        candidates = [
            e for e in elements 
            if e.element_type in (ElementType.BUTTON, ElementType.LINK)
            and e.is_visible
            and e.is_enabled
        ]
        
        # Score each candidate
        best_match: Optional[WebElement] = None
        best_score = 0
        
        for element in candidates:
            score = self._score_submit_button(element)
            if score > best_score:
                best_score = score
                best_match = element
        
        if best_match:
            logger.debug(f"Found submit button: '{best_match.text_content}' (score: {best_score})")
        else:
            logger.debug("No submit button found")
        
        return best_match
    
    def _score_submit_button(self, element: WebElement) -> int:
        """
        Score how likely an element is the submit button.
        
        Higher score = more likely to be the submit button.
        """
        score = 0
        text = element.text_content.lower()
        label = (element.aria_label or "").lower()
        attrs = element.attributes
        
        # Check text content for keywords
        for keyword in SUBMIT_KEYWORDS:
            if keyword in text or keyword in label:
                score += 10
                if keyword in ("next", "continue", "submit"):
                    score += 5  # Primary keywords get bonus
        
        # Check for common button classes
        class_attr = attrs.get("class", "").lower()
        if any(cls in class_attr for cls in ["submit", "next", "continue", "primary", "btn-primary"]):
            score += 5
        
        # Check type attribute
        type_attr = attrs.get("type", "").lower()
        if type_attr == "submit":
            score += 8
        
        # Penalize if it looks like a back/cancel button
        if any(word in text for word in ["back", "cancel", "previous", "close", "no thanks"]):
            score -= 20
        
        # Bonus for button elements
        if element.element_type == ElementType.BUTTON:
            score += 2
        
        return score
    
    async def _get_page_text(self, page: Page) -> str:
        """Extract visible text content from the page."""
        try:
            # Get text from body, excluding scripts and styles
            text = await page.evaluate("""
                () => {
                    const body = document.body;
                    const clone = body.cloneNode(true);
                    
                    // Remove script and style elements
                    clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                    
                    return clone.innerText || '';
                }
            """)
            return text.strip()
        except Exception as e:
            logger.warning(f"Failed to extract page text: {e}")
            return ""
    
    async def _extract_question_text(self, page: Page) -> str:
        """Try to extract the main question being asked."""
        try:
            # Common selectors for question text
            question_selectors = [
                "h1", "h2", "h3",
                ".question", ".question-text", ".survey-question",
                "[class*='question']",
                "legend",
                "label:first-of-type",
            ]
            
            for selector in question_selectors:
                try:
                    element = page.locator(selector).first
                    if await element.is_visible():
                        text = await element.inner_text()
                        if text and (len(text) > 10 and "?" in text or len(text) > 20):
                            return text.strip()[:500]
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            logger.debug(f"Failed to extract question text: {e}")
            return ""
    
    async def _detect_coupon(
        self,
        page: Page,
        page_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if the page contains a coupon/validation code.
        
        Returns:
            Tuple of (has_coupon, coupon_code).
        """
        page_text_lower = page_text.lower()
        
        # FIRST: Check for error states (code already used, invalid, etc.)
        for error_kw in ERROR_KEYWORDS:
            if error_kw in page_text_lower:
                logger.warning(f"Survey error detected: '{error_kw}' found in page")
                return False, None
        
        # Check for coupon keywords (more specific phrases)
        has_keywords = any(kw in page_text_lower for kw in COUPON_KEYWORDS)
        if not has_keywords:
            return False, None
        
        # Try to extract the code
        for pattern in COUPON_PATTERNS:
            matches = re.findall(pattern, page_text)
            for match in matches:
                code = match.strip()
                # Check against blacklist
                if code.upper() in COUPON_BLACKLIST:
                    logger.debug(f"Ignoring blacklisted word: {code}")
                    continue
                # Must contain at least one digit to be a valid code
                if not any(c.isdigit() for c in code):
                    logger.debug(f"Ignoring code without digits: {code}")
                    continue
                logger.info(f"Detected coupon code: {code}")
                return True, code
        
        # Check for elements that might contain the code
        code_selectors = [
            ".code", ".coupon", ".validation-code", ".coupon-code",
            "[class*='validation']", "[class*='coupon-code']",
            ".confirmation-code", ".reward-code",
        ]
        
        for selector in code_selectors:
            try:
                elements = page.locator(selector)
                count = await elements.count()
                
                for i in range(min(count, 10)):
                    text = await elements.nth(i).inner_text()
                    text = text.strip()
                    
                    # Check against blacklist
                    if text.upper() in COUPON_BLACKLIST:
                        continue
                    
                    # Must contain digits
                    if not any(c.isdigit() for c in text):
                        continue
                    
                    # Check if it matches a code pattern
                    for pattern in COUPON_PATTERNS:
                        if re.match(pattern, text):
                            logger.info(f"Detected coupon code from element: {text}")
                            return True, text
            except Exception:
                continue
        
        # We have keywords but couldn't extract valid code
        logger.warning("Coupon keywords found but no valid code extracted")
        return False, None
    
    async def _detect_error_message(self, page: Page) -> Optional[str]:
        """Detect any error messages on the page."""
        error_selectors = [
            ".error", ".error-message", ".alert-error",
            "[class*='error']", "[role='alert']",
            ".validation-error", ".form-error",
        ]
        
        for selector in error_selectors:
            try:
                element = page.locator(selector).first
                if await element.is_visible():
                    text = await element.inner_text()
                    if text.strip():
                        return text.strip()[:200]
            except Exception:
                continue
        
        return None
