"""
Test suite for the Fulfillment Agent (Module 4).

This module tests:
- Coupon detection from mock HTML pages
- Email sending with mocked SMTP
- Coupon validation patterns

Run with: pytest tests/test_fulfillment.py -v
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest

from src.survey_bot.agents.fulfillment import (
    FulfillmentAgent,
    COUPON_REGEX_PATTERNS,
)
from src.survey_bot.models.survey_result import (
    CouponResult,
    CouponPattern,
    ExtractionMethod,
    EmailDeliveryResult,
)


# =============================================================================
# MOCK RECEIPT DATA (Since ReceiptData model may not exist yet)
# =============================================================================

class MockReceiptData:
    """Mock ReceiptData for testing."""
    
    def __init__(
        self,
        store_name: str = "Test Store",
        survey_url: str = "https://survey.teststore.com/abc123",
    ):
        self.store_name = store_name
        self.survey_url = survey_url


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fulfillment_agent(tmp_path: Path) -> FulfillmentAgent:
    """Create a FulfillmentAgent with temporary screenshot directory."""
    return FulfillmentAgent(
        screenshot_dir=tmp_path / "screenshots",
        min_confidence=0.5,
    )


@pytest.fixture
def sample_coupon() -> CouponResult:
    """Create a sample CouponResult for testing."""
    return CouponResult(
        code="ABCD-123456",
        confidence=0.95,
        pattern_type=CouponPattern.ALPHA_DASH_NUMERIC,
        extraction_method=ExtractionMethod.REGEX_PATTERN,
        raw_text="Your validation code is: ABCD-123456",
    )


@pytest.fixture
def sample_receipt() -> MockReceiptData:
    """Create a sample ReceiptData for testing."""
    return MockReceiptData(
        store_name="Walmart Supercenter",
        survey_url="https://survey.walmart.com/r/ABC123",
    )


@pytest.fixture
def mock_page_with_coupon():
    """Create a mock Playwright page with coupon HTML."""
    
    async def _create_mock(coupon_code: str = "WXYZ-789012"):
        """Create mock page with given coupon code."""
        # Use MagicMock as base - only specific methods are async in Playwright
        mock_page = MagicMock()
        
        # Mock evaluate (ASYNC) to return page text
        # IMPORTANT: Avoid words that match coupon patterns (8+ alphanumeric)
        mock_page.evaluate = AsyncMock(return_value=f"""
            Done!
            
            Survey ok.
            
            Code:
            
            {coupon_code}
            
            Use it soon.
            30 days left.
        """)
        
        # Locator setup: page.locator() is SYNC, but methods are ASYNC
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=1)
        mock_locator.nth.return_value = mock_locator  # nth() is sync
        mock_locator.is_visible = AsyncMock(return_value=True)
        mock_locator.inner_text = AsyncMock(return_value=coupon_code)
        
        mock_page.locator.return_value = mock_locator
        
        # Mock screenshot (ASYNC)
        mock_page.screenshot = AsyncMock()
        
        return mock_page
    
    return _create_mock


# =============================================================================
# TEST: COUPON DETECTION FROM MOCK PAGE
# =============================================================================

class TestCouponDetectionMockPage:
    """Test coupon detection from mock HTML pages."""
    
    @pytest.mark.asyncio
    async def test_detect_standard_format_coupon(
        self,
        fulfillment_agent: FulfillmentAgent,
        mock_page_with_coupon,
    ):
        """Test detection of standard XXXX-XXXXXX format coupon."""
        mock_page = await mock_page_with_coupon("ABCD-123456")
        
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=False)
        
        assert result is not None
        assert result.code == "ABCD-123456"
        assert result.confidence >= 0.5
        assert result.pattern_type in [
            CouponPattern.ALPHA_DASH_NUMERIC,
            CouponPattern.ALPHA_NUMERIC,
        ]
    
    @pytest.mark.asyncio
    async def test_detect_numeric_only_coupon(
        self,
        fulfillment_agent: FulfillmentAgent,
        mock_page_with_coupon,
    ):
        """Test detection of numeric-only coupon (Walmart style)."""
        mock_page = await mock_page_with_coupon("123456789012")
        
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=False)
        
        assert result is not None
        assert result.code == "123456789012"
        assert result.pattern_type in [
            CouponPattern.WALMART,
            CouponPattern.NUMERIC_ONLY,
        ]
    
    @pytest.mark.asyncio
    async def test_detect_alphanumeric_coupon(
        self,
        fulfillment_agent: FulfillmentAgent,
        mock_page_with_coupon,
    ):
        """Test detection of alphanumeric coupon without dashes."""
        mock_page = await mock_page_with_coupon("SAVE20PROMO")
        
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=False)
        
        assert result is not None
        assert result.code == "SAVE20PROMO"
        assert result.pattern_type == CouponPattern.ALPHA_NUMERIC
    
    @pytest.mark.asyncio
    async def test_detect_coupon_with_screenshot(
        self,
        fulfillment_agent: FulfillmentAgent,
        mock_page_with_coupon,
    ):
        """Test that screenshot is taken when requested."""
        mock_page = await mock_page_with_coupon("TEST-123456")
        
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=True)
        
        assert result is not None
        # Screenshot method should have been called
        mock_page.screenshot.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_coupon_found(
        self,
        fulfillment_agent: FulfillmentAgent,
    ):
        """Test handling when no coupon is found on page."""
        # Use MagicMock as base - only specific methods are async
        mock_page = MagicMock()
        
        # Page with no coupon patterns - use SHORT words only (< 6 chars)
        mock_page.evaluate = AsyncMock(return_value="""
            Hi!
            
            We love you.
            Have a day!
        """)
        
        # Mock locator returns empty
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=0)
        mock_page.locator.return_value = mock_locator
        
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=False)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_detect_coupon_from_element(
        self,
        fulfillment_agent: FulfillmentAgent,
    ):
        """Test coupon detection from specific DOM element."""
        # Use MagicMock as base - only specific methods are async in Playwright
        mock_page = MagicMock()
        
        # These are the ASYNC methods on a Playwright page
        mock_page.evaluate = AsyncMock(return_value="No code in text")
        mock_page.screenshot = AsyncMock()
        
        # Create element mock with async methods
        mock_element = MagicMock()
        mock_element.is_visible = AsyncMock(return_value=True)
        mock_element.inner_text = AsyncMock(return_value="ELEM-567890")
        
        # Locator is returned sync, but its methods are async
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=1)
        mock_locator.nth.return_value = mock_element
        
        # page.locator() is SYNC - returns a Locator object
        mock_page.locator.return_value = mock_locator
        
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=False)
        
        assert result is not None
        assert result.code == "ELEM-567890"


# =============================================================================
# TEST: EMAIL SENDING WITH MOCK SMTP
# =============================================================================

class TestEmailSendMock:
    """Test email sending with mocked SMTP server."""
    
    def test_email_message_formatted_correctly(
        self,
        fulfillment_agent: FulfillmentAgent,
        sample_coupon: CouponResult,
        sample_receipt: MockReceiptData,
    ):
        """Test that email message is formatted correctly."""
        msg = fulfillment_agent._create_email_message(
            recipient="test@example.com",
            coupon=sample_coupon,
            receipt_data=sample_receipt,
            from_email="bot@example.com",
            from_name="Survey Bot",
            attach_screenshot=False,
        )
        
        # Verify message structure
        assert isinstance(msg, MIMEMultipart)
        assert msg["To"] == "test@example.com"
        assert "Survey Bot" in msg["From"]
        assert "Coupon" in msg["Subject"] or "coupon" in msg["Subject"].lower()
        
        # Verify the coupon code is in the email body
        # Get the email payload
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                assert sample_coupon.code in body
            elif part.get_content_type() == "text/html":
                html = part.get_payload(decode=True).decode()
                assert sample_coupon.code in html
    
    def test_email_contains_store_info(
        self,
        fulfillment_agent: FulfillmentAgent,
        sample_coupon: CouponResult,
        sample_receipt: MockReceiptData,
    ):
        """Test that email contains store information."""
        msg = fulfillment_agent._create_email_message(
            recipient="test@example.com",
            coupon=sample_coupon,
            receipt_data=sample_receipt,
            from_email="bot@example.com",
            from_name="Survey Bot",
            attach_screenshot=False,
        )
        
        # Check for store name in body
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                assert sample_receipt.store_name in body
    
    @pytest.mark.asyncio
    async def test_send_email_success(
        self,
        fulfillment_agent: FulfillmentAgent,
        sample_coupon: CouponResult,
        sample_receipt: MockReceiptData,
    ):
        """Test successful email sending with mocked SMTP."""
        with patch.dict('os.environ', {
            'SMTP_HOST': 'smtp.test.com',
            'SMTP_PORT': '587',
            'SMTP_USER': 'test@test.com',
            'SMTP_PASSWORD': 'testpassword',
        }):
            with patch('smtplib.SMTP') as mock_smtp_class:
                # Setup mock SMTP instance
                mock_smtp = MagicMock()
                mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
                mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
                mock_smtp.sendmail.return_value = {}
                
                result = await fulfillment_agent.send_email(
                    recipient="user@example.com",
                    coupon=sample_coupon,
                    receipt_data=sample_receipt,
                )
                
                assert result.success is True
                assert result.recipient == "user@example.com"
                assert result.coupon_code == sample_coupon.code
                
                # Verify SMTP was called
                mock_smtp_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_email_with_attachment(
        self,
        fulfillment_agent: FulfillmentAgent,
        sample_receipt: MockReceiptData,
        tmp_path: Path,
    ):
        """Test email sending with screenshot attachment."""
        # Create a screenshot file
        screenshot_path = tmp_path / "test_screenshot.png"
        screenshot_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)  # Minimal PNG header
        
        coupon_with_screenshot = CouponResult(
            code="SCREENSHOT-123",
            confidence=0.95,
            pattern_type=CouponPattern.ALPHA_DASH_NUMERIC,
            screenshot_path=str(screenshot_path),
        )
        
        msg = fulfillment_agent._create_email_message(
            recipient="test@example.com",
            coupon=coupon_with_screenshot,
            receipt_data=sample_receipt,
            from_email="bot@example.com",
            from_name="Survey Bot",
            attach_screenshot=True,
        )
        
        # Check for attachment
        has_attachment = False
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                has_attachment = True
                assert "coupon_proof.png" in part.get_filename()
        
        assert has_attachment, "Email should have screenshot attachment"
    
    @pytest.mark.asyncio
    async def test_send_email_failure_no_smtp_config(
        self,
        fulfillment_agent: FulfillmentAgent,
        sample_coupon: CouponResult,
        sample_receipt: MockReceiptData,
    ):
        """Test email failure when SMTP is not configured."""
        # Clear SMTP environment variables
        with patch.dict('os.environ', {}, clear=True):
            result = await fulfillment_agent.send_email(
                recipient="user@example.com",
                coupon=sample_coupon,
                receipt_data=sample_receipt,
            )
            
            assert result.success is False
            assert result.error_message is not None
            assert "SMTP" in result.error_message or "config" in result.error_message.lower()
    
    def test_email_validation(self, fulfillment_agent: FulfillmentAgent):
        """Test email address validation."""
        # Valid emails
        assert fulfillment_agent.validate_email("user@example.com") is True
        assert fulfillment_agent.validate_email("test.user@domain.co.uk") is True
        assert fulfillment_agent.validate_email("user+tag@example.org") is True
        
        # Invalid emails
        assert fulfillment_agent.validate_email("") is False
        assert fulfillment_agent.validate_email("notanemail") is False
        assert fulfillment_agent.validate_email("missing@domain") is False
        assert fulfillment_agent.validate_email("@nodomain.com") is False


# =============================================================================
# TEST: COUPON VALIDATION PATTERNS
# =============================================================================

class TestCouponValidationPatterns:
    """Test coupon format validation against various patterns."""
    
    @pytest.mark.parametrize("code,expected", [
        # Valid patterns
        ("ABC-1234", True),      # Short alphanumeric with dash
        ("ABCD-123456", True),   # Standard format (4 alpha, 6 numeric)
        ("12345678", True),      # 8-digit numeric
        ("SAVE20", True),        # Short alphanumeric promo code
        ("123456789012", True),  # 12-digit Walmart style
        ("WXYZ1234", True),      # 8-char alphanumeric
        ("PROMO-CODE1", True),   # Alpha-dash-alpha pattern
        ("1234-5678", True),     # Numeric with dash
        ("ABCD1234EF", True),    # 10-char alphanumeric
        
        # Invalid patterns
        ("too-long-code-here-12345", False),  # Too long
        ("!!!", False),          # Special characters only
        ("AB", False),           # Too short
        ("@#$%", False),         # Invalid characters
        ("", False),             # Empty string
        ("A" * 20, False),       # Way too long
        ("12345", False),        # 5 digits (ZIP code length - filtered)
    ])
    def test_coupon_format_validation(
        self,
        fulfillment_agent: FulfillmentAgent,
        code: str,
        expected: bool,
    ):
        """Test validation of various coupon formats."""
        result = fulfillment_agent.validate_coupon_format(code)
        assert result == expected, f"Code '{code}' should be {'valid' if expected else 'invalid'}"
    
    def test_valid_standard_codes(self, fulfillment_agent: FulfillmentAgent):
        """Test validation of standard coupon codes."""
        valid_codes = [
            "ABC-1234",       # Letters-dash-numbers
            "12345678",       # 8 digits
            "SAVE20",         # Short promo code
            "WXYZ-123456",    # Standard 4-6 format
            "ABCD1234",       # 8-char alphanumeric
        ]
        
        for code in valid_codes:
            assert fulfillment_agent.validate_coupon_format(code) is True, \
                f"Code '{code}' should be valid"
    
    def test_invalid_codes_rejected(self, fulfillment_agent: FulfillmentAgent):
        """Test that invalid codes are rejected."""
        invalid_codes = [
            "too-long-code-here-12345",  # Exceeds max length
            "!!!",                        # Special characters
            "A",                          # Too short
            "",                           # Empty
            "  ",                         # Whitespace only
            "aaaaaaaaaaaaaaaaaaaaaaaaa",  # Way too long
        ]
        
        for code in invalid_codes:
            assert fulfillment_agent.validate_coupon_format(code) is False, \
                f"Code '{code}' should be invalid"
    
    def test_pattern_identification(self, fulfillment_agent: FulfillmentAgent):
        """Test that patterns are correctly identified."""
        # Numeric only pattern
        pattern = fulfillment_agent._identify_pattern("123456789012")
        assert pattern == CouponPattern.WALMART
        
        # Alpha-dash-numeric pattern
        pattern = fulfillment_agent._identify_pattern("ABCD-123456")
        assert pattern == CouponPattern.ALPHA_DASH_NUMERIC
        
        # Pure alphanumeric pattern
        pattern = fulfillment_agent._identify_pattern("PROMO2024")
        assert pattern == CouponPattern.ALPHA_NUMERIC
        
        # Invalid pattern returns None
        pattern = fulfillment_agent._identify_pattern("!!!")
        assert pattern is None
    
    def test_false_positive_detection(self, fulfillment_agent: FulfillmentAgent):
        """Test that false positives are detected."""
        # Phone numbers should be rejected
        assert fulfillment_agent._is_false_positive(
            "8001234567",
            "Call us at 800-123-4567"
        ) is True
        
        # ZIP codes should be rejected
        assert fulfillment_agent._is_false_positive(
            "12345",
            "Located at 12345 Main St"
        ) is True
        
        # Transaction numbers should be rejected
        assert fulfillment_agent._is_false_positive(
            "9876543210",
            "Transaction number: 9876543210"
        ) is True
        
        # Real coupons should pass
        assert fulfillment_agent._is_false_positive(
            "ABCD123456",
            "Your coupon code is: ABCD123456"
        ) is False
    
    def test_code_normalization(self):
        """Test that codes are normalized correctly."""
        # CouponResult normalizes codes to uppercase
        coupon = CouponResult(
            code="  abcd-123456  ",  # Lowercase with spaces
            confidence=0.9,
        )
        
        assert coupon.code == "ABCD-123456"  # Should be uppercase and trimmed
    
    def test_display_format(self):
        """Test code display formatting."""
        # 12-digit code gets formatted as XXXX-XXXX-XXXX
        coupon = CouponResult(code="123456789012", confidence=0.9)
        assert coupon.to_display_format() == "1234-5678-9012"
        
        # 10-digit code gets formatted as XXXX-XXXXXX
        coupon = CouponResult(code="1234567890", confidence=0.9)
        assert coupon.to_display_format() == "1234-567890"
        
        # Code with dashes stays as-is
        coupon = CouponResult(code="ABCD-123456", confidence=0.9)
        formatted = coupon.to_display_format()
        # Either keeps dashes or reformats consistently
        assert "-" in formatted or len(formatted) == 10


# =============================================================================
# INTEGRATION-STYLE TESTS (Still use mocks but test full flow)
# =============================================================================

class TestFulfillmentIntegration:
    """Integration-style tests for the full fulfillment flow."""
    
    @pytest.mark.asyncio
    async def test_full_detection_and_validation_flow(
        self,
        fulfillment_agent: FulfillmentAgent,
        mock_page_with_coupon,
    ):
        """Test the full flow from detection to validation."""
        mock_page = await mock_page_with_coupon("VALID-123456")
        
        # Detect coupon
        result = await fulfillment_agent.detect_coupon(mock_page, take_screenshot=False)
        
        assert result is not None
        
        # Validate the detected code
        is_valid = fulfillment_agent.validate_coupon_format(result.code)
        assert is_valid is True
        
        # Identify pattern
        pattern = fulfillment_agent._identify_pattern(result.code)
        assert pattern is not None
    
    def test_email_template_completeness(
        self,
        fulfillment_agent: FulfillmentAgent,
        sample_coupon: CouponResult,
        sample_receipt: MockReceiptData,
    ):
        """Test that email templates contain all required information."""
        # Plain text format
        plain_text = fulfillment_agent.format_email_body(sample_coupon, sample_receipt)
        
        assert sample_coupon.code in plain_text
        assert sample_receipt.store_name in plain_text
        assert "expire" in plain_text.lower() or "days" in plain_text.lower()
        
        # HTML format
        html = fulfillment_agent.format_email_body_html(sample_coupon, sample_receipt)
        
        assert sample_coupon.code in html
        assert sample_receipt.store_name in html
        assert "<html>" in html.lower()
        assert "<!doctype" in html.lower() or "<body" in html.lower()
    
    def test_get_smtp_config(self, fulfillment_agent: FulfillmentAgent):
        """Test SMTP configuration retrieval."""
        with patch.dict('os.environ', {
            'SMTP_HOST': 'smtp.example.com',
            'SMTP_PORT': '465',
            'SMTP_USER': 'user@example.com',
            'SMTP_PASSWORD': 'secret',
            'SMTP_FROM_NAME': 'My Bot',
        }):
            config = fulfillment_agent.get_smtp_config()
            
            assert config["host"] == "smtp.example.com"
            assert config["port"] == "465"
            assert config["user"] == "user@example.com"
            assert config["password"] == "****"  # Should be masked
            assert config["from_name"] == "My Bot"
    
    def test_pattern_description(self, fulfillment_agent: FulfillmentAgent):
        """Test human-readable pattern descriptions."""
        desc = fulfillment_agent.get_pattern_description(CouponPattern.WALMART)
        assert "12" in desc or "Walmart" in desc or "digit" in desc.lower()
        
        desc = fulfillment_agent.get_pattern_description(CouponPattern.ALPHA_DASH_NUMERIC)
        assert "dash" in desc.lower() or "-" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
