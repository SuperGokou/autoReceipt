"""
Fulfillment Agent for coupon extraction and delivery.

This module provides the FulfillmentAgent class which handles:
- Detecting coupon/validation codes on completion pages
- Validating coupon format against known patterns
- Taking proof screenshots
- Sending coupons via email

The agent is the final step in the survey automation pipeline,
responsible for extracting the reward and delivering it to the user.

Configuration for email delivery via environment variables:
- SMTP_HOST: SMTP server hostname (e.g., smtp.gmail.com)
- SMTP_PORT: SMTP server port (e.g., 587 for TLS)
- SMTP_USER: SMTP username/email
- SMTP_PASSWORD: SMTP password or app password
- SMTP_FROM_NAME: Sender display name (optional)

Example Usage:
    >>> from playwright.async_api import Page
    >>> from survey_bot.agents.fulfillment import FulfillmentAgent
    >>> 
    >>> agent = FulfillmentAgent()
    >>> result = await agent.detect_coupon(page)
    >>> 
    >>> if result:
    ...     print(f"Found coupon: {result.code}")
    ...     # Send via email
    ...     success = await agent.send_email(
    ...         recipient="user@example.com",
    ...         coupon=result,
    ...         receipt_data=receipt
    ...     )
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
import ssl
from datetime import datetime
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Tuple

from ..models.receipt import ReceiptData
from ..models.survey_result import (
    CouponPattern,
    CouponResult,
    EmailDeliveryResult,
    ExtractionMethod,
)


if TYPE_CHECKING:
    from playwright.async_api import Page, Locator


__all__ = ["FulfillmentAgent"]

logger = logging.getLogger(__name__)


# =============================================================================
# COUPON DETECTION PATTERNS
# =============================================================================

# Regex patterns for common coupon formats
COUPON_REGEX_PATTERNS: List[Tuple[str, CouponPattern, float]] = [
    # Pattern, Type, Confidence
    
    # XXXX-XXXXXX (4 letters, dash, 6 digits) - Very common
    (r'\b([A-Z]{4}[-\s]?\d{6})\b', CouponPattern.ALPHA_DASH_NUMERIC, 0.95),
    
    # XXXXXX-XXXX (6 digits, dash, 4 letters)
    (r'\b(\d{6}[-\s]?[A-Z]{4})\b', CouponPattern.ALPHA_DASH_NUMERIC, 0.95),
    
    # XXXX-XXXX-XXXX (3 groups of 4)
    (r'\b([A-Z0-9]{4}[-\s][A-Z0-9]{4}[-\s][A-Z0-9]{4})\b', CouponPattern.ALPHA_DASH_NUMERIC, 0.90),
    
    # 12 digit numeric (Walmart style)
    (r'\b(\d{12})\b', CouponPattern.WALMART, 0.85),
    
    # 10-11 digit numeric
    (r'\b(\d{10,11})\b', CouponPattern.NUMERIC_ONLY, 0.80),
    
    # XXXXX-XXXXX (5-5 format)
    (r'\b([A-Z0-9]{5}[-\s][A-Z0-9]{5})\b', CouponPattern.ALPHA_DASH_NUMERIC, 0.85),
    
    # Pure alphanumeric 8-12 chars
    (r'\b([A-Z0-9]{8,12})\b', CouponPattern.ALPHA_NUMERIC, 0.70),
    
    # 6-8 digit numeric (like McDonald's validation codes)
    (r'\b(\d{6,8})\b', CouponPattern.NUMERIC_ONLY, 0.75),
]

# Blacklist of common English words that look like codes but aren't
COUPON_BLACKLIST = {
    # Common survey words
    "COMPLETING", "COMPLETE", "COMPLETED",
    "SURVEY", "SURVEYS", "SURVEYED",
    "FEEDBACK", "CUSTOMER", "SATISFACTION",
    "THANK", "THANKS", "THANKYOU",
    "PLEASE", "WELCOME", "VALUED",
    "RECEIPT", "RECEIPTS", "PURCHASE",
    "RESTAURANT", "LOCATION", "STORE",
    "MCDONALDS", "MCDONALD", "WENDYS", "SUBWAY",
    "PANDAEXPRESS", "TACOBELL", "CHICKFILA",
    "SERVICE", "EXPERIENCE", "QUALITY",
    "MANAGER", "EMPLOYEE", "STAFF",
    "VALIDITY", "VALIDATION", "VALIDATE",
    "REQUIRED", "OPTIONAL", "IMPORTANT",
    "PRIVACY", "POLICY", "TERMS",
    "PARTICIPATING", "LOCATIONS", "VISIT",
    "VISITING", "VISITED",
    "REDEEM", "REDEEMED", "REDEMPTION",
    "EXPIRES", "EXPIRED", "EXPIRATION",
    "INVALID", "VALID", "VERIFY",
    "SUBMIT", "SUBMITTED", "CONTINUE",
    "WEBSITE", "ONLINE", "INTERNET",
    "CONTACT", "SUPPORT", "HELP",
    # More common words
    "APPRECIATE", "CANDID", "TAKING", "TIME",
    "START", "STARTED", "BEGIN", "FINISH",
    "ENGLISH", "SPANISH", "ESPANOL",
    "VERSION", "FRIENDLY", "ACCESSIBILITY",
    "RESERVED", "RIGHTS", "COPYRIGHT",
    "POWERED", "GROUP", "MANAGEMENT",
    # Common marketing phrases that get misdetected
    "HEARMOREFROM", "HEARMORE", "TELLUS", "TELLUSABOUT",
    "SHAREMORE", "LEARNMORE", "FINDOUT", "GETMORE",
    "STAYTUNE", "STAYTUNED", "FOLLOWING", "SUBSCRIBE",
    # Email page words
    "INDICATES", "INFORMATION", "ADDRESS", "CONFIRM",
    "PROVIDE", "RECEIVE", "PURPOSE", "CLICK",
    "EMAIL", "EMAILS", "COUPON", "CODE",
    # Thank you page words (no code displayed)
    "CONNECTED", "REWARDS", "FORWARD", "SERVING",
    "SOON", "AGAIN", "STAY", "LOOKING", "PANDA",
}

# Keywords that indicate coupon/code presence
COUPON_KEYWORDS = [
    "validation code",
    "validation number",
    "coupon code",
    "coupon",
    "voucher",
    "discount code",
    "promo code",
    "redemption code",
    "confirmation code",
    "survey code",
    "code:",
    "your code",
    "reward code",
    "gift code",
]

# CSS selectors for common coupon display elements
COUPON_ELEMENT_SELECTORS = [
    # Specific class names
    ".coupon-code",
    ".validation-code",
    ".code",
    ".coupon",
    ".voucher",
    ".promo-code",
    ".reward-code",
    
    # Class patterns
    "[class*='code']",
    "[class*='coupon']",
    "[class*='validation']",
    "[class*='voucher']",
    
    # Data attributes
    "[data-coupon]",
    "[data-code]",
    "[data-validation]",
    
    # Common styling for codes
    ".highlight",
    ".emphasized",
    ".large-text",
    
    # Typography elements that might contain codes
    "h1", "h2", "h3",
    "strong", "b",
    ".big", ".large",
]

# Default screenshot directory
DEFAULT_SCREENSHOT_DIR = Path("screenshots")


class FulfillmentAgent:
    """
    Agent for extracting and delivering survey rewards.
    
    Handles the final stage of survey automation:
    1. Detect coupon/validation codes on completion pages
    2. Validate code format
    3. Capture proof screenshots
    4. Send codes via email (optional)
    
    Attributes:
        screenshot_dir: Directory for saving proof screenshots.
        min_confidence: Minimum confidence threshold for codes.
    """
    
    def __init__(
        self,
        screenshot_dir: Optional[Path] = None,
        min_confidence: float = 0.5,
    ) -> None:
        """
        Initialize the FulfillmentAgent.
        
        Args:
            screenshot_dir: Directory for screenshots (created if needed).
            min_confidence: Minimum confidence for accepting codes.
        """
        self.screenshot_dir = screenshot_dir or DEFAULT_SCREENSHOT_DIR
        self.min_confidence = min_confidence
        
        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(
            f"FulfillmentAgent initialized: screenshot_dir={self.screenshot_dir}"
        )
    
    async def detect_coupon(
        self,
        page: "Page",
        take_screenshot: bool = True,
    ) -> Optional[CouponResult]:
        """
        Detect and extract coupon code from the current page.
        
        Uses multiple strategies:
        1. Search for elements with coupon-related classes/attributes
        2. Look for text matching known coupon patterns
        3. Find emphasized/highlighted text that looks like codes
        
        Args:
            page: Playwright Page object (should be on completion page).
            take_screenshot: Whether to capture proof screenshot.
            
        Returns:
            CouponResult if found, None otherwise.
        """
        logger.info("Detecting coupon on page...")
        
        # Strategy 1: Look for specific coupon elements
        result = await self._detect_from_elements(page)
        if result and result.confidence >= self.min_confidence:
            if take_screenshot:
                result.screenshot_path = await self.take_coupon_screenshot(page, result.code)
            logger.info(f"Found coupon via element search: {result.code}")
            return result
        
        # Strategy 2: Search page text with regex patterns
        result = await self._detect_from_page_text(page)
        if result and result.confidence >= self.min_confidence:
            if take_screenshot:
                result.screenshot_path = await self.take_coupon_screenshot(page, result.code)
            logger.info(f"Found coupon via text pattern: {result.code}")
            return result
        
        # Strategy 3: Look for emphasized text near keywords
        result = await self._detect_from_context(page)
        if result and result.confidence >= self.min_confidence:
            if take_screenshot:
                result.screenshot_path = await self.take_coupon_screenshot(page, result.code)
            logger.info(f"Found coupon via context: {result.code}")
            return result
        
        logger.warning("No coupon code detected on page")
        return None
    
    async def _detect_from_elements(self, page: "Page") -> Optional[CouponResult]:
        """
        Search for coupon in specific DOM elements.
        
        Looks for elements with coupon-related classes, IDs, or attributes.
        """
        for selector in COUPON_ELEMENT_SELECTORS:
            try:
                elements = page.locator(selector)
                count = await elements.count()
                
                for i in range(min(count, 10)):  # Check up to 10 elements
                    element = elements.nth(i)
                    
                    if not await element.is_visible():
                        continue
                    
                    text = await element.inner_text()
                    text = text.strip()
                    
                    # Try to extract code from text
                    code = self._extract_code_from_text(text)
                    if code:
                        # Validate the code
                        pattern = self._identify_pattern(code)
                        if pattern:
                            return CouponResult(
                                code=code,
                                confidence=0.90,
                                pattern_type=pattern,
                                extraction_method=ExtractionMethod.ELEMENT_SEARCH,
                                raw_text=text[:100],
                            )
            except Exception as e:
                logger.debug(f"Error checking selector {selector}: {e}")
                continue
        
        return None
    
    async def _detect_from_page_text(self, page: "Page") -> Optional[CouponResult]:
        """
        Search for coupon in full page text using regex patterns.
        """
        try:
            # Get page text content
            page_text = await page.evaluate("""
                () => {
                    const body = document.body;
                    const clone = body.cloneNode(true);
                    clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                    return clone.innerText || '';
                }
            """)
            
            # Search with each pattern
            for pattern_str, pattern_type, confidence in COUPON_REGEX_PATTERNS:
                matches = re.findall(pattern_str, page_text.upper())
                
                if matches:
                    # Get the first valid match
                    for match in matches:
                        code = match.replace(" ", "").replace("-", "-")
                        
                        # Skip if too short or doesn't look like a code
                        if len(code.replace("-", "")) < 6:
                            continue
                        
                        # Skip common false positives
                        if self._is_false_positive(code, page_text):
                            continue
                        
                        return CouponResult(
                            code=code,
                            confidence=confidence,
                            pattern_type=pattern_type,
                            extraction_method=ExtractionMethod.REGEX_PATTERN,
                            raw_text=page_text[:200],
                        )
            
        except Exception as e:
            logger.warning(f"Error extracting from page text: {e}")
        
        return None
    
    async def _detect_from_context(self, page: "Page") -> Optional[CouponResult]:
        """
        Search for code near coupon-related keywords.
        
        Looks for emphasized text that appears near keywords like
        "validation code", "coupon", etc.
        """
        try:
            # Look for keywords and nearby emphasized text
            for keyword in COUPON_KEYWORDS:
                try:
                    # Find elements containing the keyword
                    keyword_elements = page.locator(f"text=/{keyword}/i")
                    count = await keyword_elements.count()
                    
                    if count == 0:
                        continue
                    
                    # Check nearby siblings and children
                    for i in range(min(count, 5)):
                        element = keyword_elements.nth(i)
                        
                        # Get parent context
                        parent = element.locator("xpath=..")
                        parent_text = await parent.inner_text()
                        
                        # Look for code in parent text
                        code = self._extract_code_from_text(parent_text)
                        if code:
                            pattern = self._identify_pattern(code)
                            if pattern:
                                return CouponResult(
                                    code=code,
                                    confidence=0.75,
                                    pattern_type=pattern,
                                    extraction_method=ExtractionMethod.ELEMENT_SEARCH,
                                    raw_text=parent_text[:100],
                                )
                        
                        # Check following sibling
                        sibling = element.locator("xpath=following-sibling::*[1]")
                        if await sibling.count() > 0:
                            sibling_text = await sibling.inner_text()
                            code = self._extract_code_from_text(sibling_text)
                            if code:
                                pattern = self._identify_pattern(code)
                                if pattern:
                                    return CouponResult(
                                        code=code,
                                        confidence=0.80,
                                        pattern_type=pattern,
                                        extraction_method=ExtractionMethod.ELEMENT_SEARCH,
                                        raw_text=sibling_text[:100],
                                    )
                
                except Exception:
                    continue
            
        except Exception as e:
            logger.warning(f"Error in context detection: {e}")
        
        return None
    
    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """
        Extract a coupon code from arbitrary text.
        
        Args:
            text: Text that may contain a coupon code.
            
        Returns:
            Extracted code or None.
        """
        if not text:
            return None
        
        text_upper = text.upper()
        
        # Priority 1: Look for explicit "Validation Code: XXXX" pattern
        validation_patterns = [
            r'VALIDATION\s*CODE[:\s]+(\d{6,12})',
            r'VALIDATION\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'COUPON\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'YOUR\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'REDEMPTION\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'CODE[:\s]+(\d{6,12})\b',
        ]
        
        for pattern in validation_patterns:
            match = re.search(pattern, text_upper)
            if match:
                code = match.group(1).strip()
                if code and code not in COUPON_BLACKLIST:
                    return code
        
        # Priority 2: Try each general pattern
        codes_with_digits = []
        codes_without_digits = []

        for pattern_str, _, _ in COUPON_REGEX_PATTERNS:
            matches = re.findall(pattern_str, text_upper)
            for code in matches:
                code = code.replace(" ", "")
                code_clean = code.replace("-", "")

                # Skip blacklisted words
                if code in COUPON_BLACKLIST or code_clean in COUPON_BLACKLIST:
                    continue

                # Must be at least 6 chars
                if len(code_clean) >= 6:
                    # Separate codes with and without digits
                    if any(c.isdigit() for c in code):
                        codes_with_digits.append(code)
                    else:
                        codes_without_digits.append(code)

        # STRONGLY prefer codes with digits (validation codes almost always have numbers)
        if codes_with_digits:
            return codes_with_digits[0]

        # Only return pure alpha codes if no numeric codes found
        # This prevents false positives like "HEARMOREFROM"
        if codes_without_digits:
            logger.warning(f"Found non-numeric code candidates: {codes_without_digits}, may be false positive")
            return None  # Don't return pure alpha codes - too many false positives

        return None
    
    def _identify_pattern(self, code: str) -> Optional[CouponPattern]:
        """
        Identify which coupon pattern a code matches.
        
        Args:
            code: The coupon code to check.
            
        Returns:
            CouponPattern if valid, None otherwise.
        """
        if not self.validate_coupon_format(code):
            return None
        
        code_clean = code.replace("-", "").replace(" ", "")
        
        # Check specific patterns
        if code_clean.isdigit():
            if len(code_clean) == 12:
                return CouponPattern.WALMART
            return CouponPattern.NUMERIC_ONLY
        
        if "-" in code or len(code) > len(code_clean):
            return CouponPattern.ALPHA_DASH_NUMERIC
        
        return CouponPattern.ALPHA_NUMERIC
    
    def _is_false_positive(self, code: str, context: str) -> bool:
        """
        Check if a potential code is likely a false positive.
        
        Args:
            code: The potential coupon code.
            context: Surrounding text for context.
            
        Returns:
            True if likely false positive.
        """
        code_clean = code.replace("-", "").replace(" ", "").upper()
        
        # Check blacklist
        if code.upper() in COUPON_BLACKLIST or code_clean in COUPON_BLACKLIST:
            return True
        
        # Skip phone numbers (10 digits starting with specific patterns)
        if code_clean.isdigit() and len(code_clean) == 10:
            if code_clean[:3] in ["800", "888", "877", "866", "855", "844"]:
                return True
        
        # Skip dates (MMDDYYYY, YYYYMMDD)
        if code_clean.isdigit() and len(code_clean) == 8:
            # Check if it could be a date
            try:
                year = int(code_clean[:4])
                if 1900 < year < 2100:
                    return True
                year = int(code_clean[4:])
                if 1900 < year < 2100:
                    return True
            except ValueError:
                pass
        
        # Skip ZIP codes
        if code_clean.isdigit() and len(code_clean) == 5:
            return True
        
        # Skip transaction numbers if context suggests it
        if "transaction" in context.lower() and code_clean.isdigit():
            return True
        
        # Skip store numbers
        if "store" in context.lower() and len(code_clean) <= 6:
            return True
        
        return False
    
    def validate_coupon_format(self, text: str) -> bool:
        """
        Validate if text matches a known coupon format.
        
        Checks against common coupon patterns used by survey providers.
        Most surveys use alphanumeric codes of 6-12 characters,
        optionally with dashes.
        
        Args:
            text: Text to validate as a coupon code.
            
        Returns:
            True if text matches a valid coupon format.
        """
        if not text:
            return False
        
        # Normalize
        text = text.strip().upper()
        clean_text = text.replace("-", "").replace(" ", "")
        
        # Check length (6-16 characters)
        if not 6 <= len(clean_text) <= 16:
            return False
        
        # Must be alphanumeric
        if not clean_text.isalnum():
            return False
        
        # Should not be all zeros or all same character
        if len(set(clean_text)) <= 1:
            return False
        
        # Check against patterns
        for pattern_str, _, _ in COUPON_REGEX_PATTERNS:
            if re.match(pattern_str, text):
                return True
        
        # If alphanumeric and right length, probably valid
        if clean_text.isalnum() and 6 <= len(clean_text) <= 12:
            return True
        
        return False
    
    async def take_coupon_screenshot(
        self,
        page: "Page",
        code: str,
    ) -> str:
        """
        Take a screenshot of the coupon/completion page.
        
        Args:
            page: Playwright Page object.
            code: The coupon code (for filename).
            
        Returns:
            Path to the saved screenshot.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_code = "".join(c if c.isalnum() else "_" for c in code[:10])
        filename = f"coupon_{safe_code}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        
        try:
            await page.screenshot(path=str(filepath), full_page=True)
            logger.info(f"Coupon screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""
    
    async def extract_coupon_element_screenshot(
        self,
        page: "Page",
        selector: str,
    ) -> Optional[str]:
        """
        Take a screenshot of just the coupon element.
        
        Args:
            page: Playwright Page object.
            selector: CSS selector for the coupon element.
            
        Returns:
            Path to screenshot or None.
        """
        try:
            element = page.locator(selector).first
            if await element.is_visible():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"coupon_element_{timestamp}.png"
                filepath = self.screenshot_dir / filename
                
                await element.screenshot(path=str(filepath))
                return str(filepath)
        except Exception as e:
            logger.warning(f"Could not screenshot element: {e}")
        
        return None
    
    def get_pattern_description(self, pattern: CouponPattern) -> str:
        """
        Get a human-readable description of a coupon pattern.
        
        Args:
            pattern: The coupon pattern type.
            
        Returns:
            Description string.
        """
        descriptions = {
            CouponPattern.ALPHA_NUMERIC: "Alphanumeric code (e.g., ABC123XYZ)",
            CouponPattern.ALPHA_DASH_NUMERIC: "Code with dashes (e.g., ABCD-123456)",
            CouponPattern.NUMERIC_ONLY: "Numeric code (e.g., 123456789)",
            CouponPattern.WALMART: "Walmart 12-digit code",
            CouponPattern.MCDONALDS: "McDonald's survey code",
            CouponPattern.TACO_BELL: "Taco Bell validation code",
            CouponPattern.WENDYS: "Wendy's survey code",
            CouponPattern.GENERIC: "Generic coupon code",
        }
        return descriptions.get(pattern, "Unknown format")
    
    # =========================================================================
    # EMAIL DELIVERY METHODS
    # =========================================================================
    
    async def send_email(
        self,
        recipient: str,
        coupon: CouponResult,
        receipt_data: ReceiptData,
        attach_screenshot: bool = True,
    ) -> EmailDeliveryResult:
        """
        Send coupon code via email with optional screenshot attachment.
        
        Uses SMTP with TLS for secure email delivery. Configuration is
        loaded from environment variables.
        
        Required Environment Variables:
            SMTP_HOST: SMTP server hostname (e.g., smtp.gmail.com)
            SMTP_PORT: SMTP server port (e.g., 587)
            SMTP_USER: SMTP username/email address
            SMTP_PASSWORD: SMTP password or app-specific password
        
        Optional Environment Variables:
            SMTP_FROM_NAME: Sender display name
        
        Args:
            recipient: Email address to send coupon to.
            coupon: CouponResult with code and screenshot path.
            receipt_data: ReceiptData with store info.
            attach_screenshot: Whether to attach the screenshot.
            
        Returns:
            EmailDeliveryResult with success status and details.
            
        Example:
            >>> result = await agent.send_email(
            ...     recipient="user@example.com",
            ...     coupon=coupon_result,
            ...     receipt_data=receipt
            ... )
            >>> if result.success:
            ...     print(f"Email sent! Message ID: {result.message_id}")
        """
        logger.info(f"Sending coupon email to {recipient}")
        
        # Load SMTP configuration from environment
        smtp_host = os.environ.get("SMTP_HOST")
        smtp_port = os.environ.get("SMTP_PORT", "587")
        smtp_user = os.environ.get("SMTP_USER")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        from_name = os.environ.get("SMTP_FROM_NAME", "Survey Bot")
        
        # Validate configuration
        if not all([smtp_host, smtp_user, smtp_password]):
            error_msg = (
                "SMTP configuration incomplete. Required environment variables: "
                "SMTP_HOST, SMTP_USER, SMTP_PASSWORD"
            )
            logger.error(error_msg)
            return EmailDeliveryResult(
                success=False,
                recipient=recipient,
                error_message=error_msg,
                coupon_code=coupon.code,
            )

        # Type narrowing: assert that required vars are not None
        assert smtp_host is not None
        assert smtp_user is not None
        assert smtp_password is not None

        try:
            # Create email message
            msg = self._create_email_message(
                recipient=recipient,
                coupon=coupon,
                receipt_data=receipt_data,
                from_email=smtp_user,
                from_name=from_name,
                attach_screenshot=attach_screenshot,
            )
            
            # Send email via SMTP with TLS
            context = ssl.create_default_context()
            
            with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(smtp_user, smtp_password)
                
                # Send the message
                server.send_message(msg)
                
                # Get message ID if available
                message_id = msg.get("Message-ID", "")
            
            logger.info(f"Email sent successfully to {recipient}")
            
            return EmailDeliveryResult(
                success=True,
                recipient=recipient,
                message_id=message_id,
                coupon_code=coupon.code,
            )
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {e}"
            logger.error(error_msg)
            return EmailDeliveryResult(
                success=False,
                recipient=recipient,
                error_message=error_msg,
                coupon_code=coupon.code,
            )
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {e}"
            logger.error(error_msg)
            return EmailDeliveryResult(
                success=False,
                recipient=recipient,
                error_message=error_msg,
                coupon_code=coupon.code,
            )
        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            logger.error(error_msg)
            return EmailDeliveryResult(
                success=False,
                recipient=recipient,
                error_message=error_msg,
                coupon_code=coupon.code,
            )
    
    def _create_email_message(
        self,
        recipient: str,
        coupon: CouponResult,
        receipt_data: ReceiptData,
        from_email: str,
        from_name: str,
        attach_screenshot: bool,
    ) -> MIMEMultipart:
        """
        Create the email message with HTML body and optional attachment.
        
        Args:
            recipient: Recipient email address.
            coupon: CouponResult with code.
            receipt_data: ReceiptData with store info.
            from_email: Sender email address.
            from_name: Sender display name.
            attach_screenshot: Whether to attach screenshot.
            
        Returns:
            Configured MIMEMultipart message.
        """
        # Create message container
        msg = MIMEMultipart("mixed")
        
        # Set headers
        store_name = receipt_data.store_name or "Survey"
        msg["Subject"] = f"Your {store_name} Survey Coupon Code"
        msg["From"] = f"{from_name} <{from_email}>"
        msg["To"] = recipient
        
        # Create the email body
        body_text = self.format_email_body(coupon, receipt_data)
        body_html = self.format_email_body_html(coupon, receipt_data)
        
        # Create alternative container for text/html
        msg_alternative = MIMEMultipart("alternative")
        
        # Attach plain text version
        part_text = MIMEText(body_text, "plain", "utf-8")
        msg_alternative.attach(part_text)
        
        # Attach HTML version
        part_html = MIMEText(body_html, "html", "utf-8")
        msg_alternative.attach(part_html)
        
        msg.attach(msg_alternative)
        
        # Attach screenshot if available and requested
        if attach_screenshot and coupon.screenshot_path:
            screenshot_path = Path(coupon.screenshot_path)
            if screenshot_path.exists():
                try:
                    with open(screenshot_path, "rb") as f:
                        attachment = MIMEBase("image", "png")
                        attachment.set_payload(f.read())
                        encoders.encode_base64(attachment)
                        attachment.add_header(
                            "Content-Disposition",
                            f"attachment; filename=coupon_proof.png"
                        )
                        msg.attach(attachment)
                        logger.debug(f"Attached screenshot: {screenshot_path}")
                except Exception as e:
                    logger.warning(f"Could not attach screenshot: {e}")
        
        return msg
    
    def format_email_body(
        self,
        coupon: CouponResult,
        receipt_data: ReceiptData,
    ) -> str:
        """
        Format the email body as plain text.
        
        Creates a professional email template with the coupon code
        prominently displayed.
        
        Args:
            coupon: CouponResult with code and metadata.
            receipt_data: ReceiptData with store information.
            
        Returns:
            Formatted plain text email body.
        """
        store_name = receipt_data.store_name or "the store"
        survey_url = receipt_data.survey_url or "the survey"
        extracted_time = coupon.extracted_at.strftime("%B %d, %Y at %I:%M %p")
        
        # Build the email body
        body = f"""
========================================
        YOUR SURVEY COUPON CODE
========================================

Thank you for completing the {store_name} customer survey!

Your validation code is:

    ╔══════════════════════════════════╗
    ║                                  ║
    ║          {coupon.code:^20}        ║
    ║                                  ║
    ╚══════════════════════════════════╝

----------------------------------------
DETAILS
----------------------------------------

Store:          {store_name}
Survey URL:     {survey_url}
Code Extracted: {extracted_time}
Code Format:    {self.get_pattern_description(CouponPattern(coupon.pattern_type))}

----------------------------------------
IMPORTANT REMINDERS
----------------------------------------

• Write this code on your receipt
• Most codes expire within 7-30 days
• Check your receipt for specific offer details
• Present the code on your next visit

----------------------------------------

This coupon was automatically extracted by Survey Bot.
If you have any issues, please contact support.

Happy saving!
"""
        return body.strip()
    
    def format_email_body_html(
        self,
        coupon: CouponResult,
        receipt_data: ReceiptData,
    ) -> str:
        """
        Format the email body as HTML.
        
        Creates a professional HTML email template with styled
        coupon display.
        
        Args:
            coupon: CouponResult with code and metadata.
            receipt_data: ReceiptData with store information.
            
        Returns:
            Formatted HTML email body.
        """
        store_name = receipt_data.store_name or "the store"
        survey_url = receipt_data.survey_url or "the survey"
        extracted_time = coupon.extracted_at.strftime("%B %d, %Y at %I:%M %p")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Survey Coupon Code</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    
    <!-- Header -->
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 24px;">Your Survey Coupon Code</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Thank you for completing the {store_name} survey!</p>
    </div>
    
    <!-- Coupon Code Box -->
    <div style="background: #f8f9fa; padding: 30px; text-align: center; border: 2px dashed #667eea;">
        <p style="margin: 0 0 10px 0; color: #666; font-size: 14px;">YOUR VALIDATION CODE</p>
        <div style="background: white; border: 3px solid #28a745; border-radius: 8px; padding: 20px; display: inline-block; min-width: 200px;">
            <span style="font-size: 32px; font-weight: bold; letter-spacing: 3px; color: #28a745; font-family: 'Courier New', monospace;">
                {coupon.code}
            </span>
        </div>
        <p style="margin: 15px 0 0 0; color: #666; font-size: 12px;">
            Write this code on your receipt
        </p>
    </div>
    
    <!-- Details Section -->
    <div style="background: white; padding: 25px; border: 1px solid #e9ecef;">
        <h2 style="color: #333; font-size: 18px; margin: 0 0 15px 0; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
            Details
        </h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px 0; color: #666; width: 140px;">Store:</td>
                <td style="padding: 8px 0; font-weight: bold;">{store_name}</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; color: #666;">Code Extracted:</td>
                <td style="padding: 8px 0;">{extracted_time}</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; color: #666;">Code Format:</td>
                <td style="padding: 8px 0;">{self.get_pattern_description(CouponPattern(coupon.pattern_type))}</td>
            </tr>
        </table>
    </div>
    
    <!-- Important Reminders -->
    <div style="background: #fff3cd; padding: 20px; border-left: 4px solid #ffc107; margin: 20px 0;">
        <h3 style="color: #856404; margin: 0 0 10px 0; font-size: 16px;">Important Reminders</h3>
        <ul style="margin: 0; padding-left: 20px; color: #856404;">
            <li>Write this code on your receipt</li>
            <li>Most codes expire within 7-30 days</li>
            <li>Check your receipt for specific offer details</li>
            <li>Present the code on your next visit</li>
        </ul>
    </div>
    
    <!-- Footer -->
    <div style="text-align: center; padding: 20px; color: #999; font-size: 12px; border-top: 1px solid #e9ecef;">
        <p style="margin: 0 0 10px 0;">
            This coupon was automatically extracted by <strong>Survey Bot</strong>
        </p>
        <p style="margin: 0; color: #ccc;">
            If you did not request this email, please ignore it.
        </p>
    </div>
    
</body>
</html>
"""
        return html.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate an email address format.

        Args:
            email: Email address to validate.

        Returns:
            True if email format is valid.
        """
        if not email:
            return False

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def get_smtp_config(self) -> dict:
        """
        Get current SMTP configuration from environment.
        
        Returns:
            Dictionary with SMTP settings (password masked).
        """
        return {
            "host": os.environ.get("SMTP_HOST", "(not set)"),
            "port": os.environ.get("SMTP_PORT", "587"),
            "user": os.environ.get("SMTP_USER", "(not set)"),
            "password": "****" if os.environ.get("SMTP_PASSWORD") else "(not set)",
            "from_name": os.environ.get("SMTP_FROM_NAME", "Survey Bot"),
        }
