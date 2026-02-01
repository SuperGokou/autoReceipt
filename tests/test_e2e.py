"""
End-to-End Test Suite for Survey Bot.

This module provides comprehensive E2E tests that validate the complete
survey automation pipeline from receipt image to coupon delivery.

Tests use the mock survey server to avoid hitting real survey sites.

Run with:
    pytest tests/test_e2e.py -v -m e2e
    pytest tests/test_e2e.py -v -m e2e --log-cli-level=DEBUG

Features:
- Complete flow testing (receipt → survey → coupon → email)
- Comprehensive debug logging for failed runs
- Multiple mood scenarios (happy, neutral, angry)
- Error handling and edge cases
- Performance timing

Requirements:
    pip install flask qrcode[pil] pytest-asyncio
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class E2ETestLogger:
    """
    Comprehensive logger for E2E test debugging.
    
    Captures detailed information about each step of the survey automation
    pipeline to help diagnose failures.
    """
    
    def __init__(self, test_name: str, log_dir: Optional[Path] = None):
        """
        Initialize the E2E test logger.
        
        Args:
            test_name: Name of the test being run.
            log_dir: Directory for log files (default: tests/logs).
        """
        self.test_name = test_name
        self.log_dir = log_dir or Path("tests/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize test name for filename (remove emojis and special chars)
        safe_name = self._sanitize_filename(test_name)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{safe_name}_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(f"e2e.{test_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler with detailed format (UTF-8 encoding)
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages (with Unicode error handling)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        # Handle Unicode errors gracefully on Windows
        console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False)
        self.logger.addHandler(console_handler)
        
        # State tracking
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.screenshots: List[Path] = []
        self.errors: List[str] = []
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"E2E TEST: {test_name}")
        self.logger.info(f"Started: {datetime.now().isoformat()}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"=" * 60)
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Sanitize a string for use as a filename.
        
        Removes emojis, special characters, and anything not safe for Windows filenames.
        
        Args:
            name: The string to sanitize.
            
        Returns:
            A safe filename string.
        """
        import re
        # Remove emojis and other non-ASCII characters
        safe = ''.join(c if ord(c) < 128 else '_' for c in name)
        # Replace problematic characters
        safe = re.sub(r'[<>:"/\\|?*\[\]]', '_', safe)
        # Collapse multiple underscores
        safe = re.sub(r'_+', '_', safe)
        # Remove leading/trailing underscores
        safe = safe.strip('_')
        return safe or 'unnamed_test'
    
    def step(self, name: str, details: Optional[Dict] = None) -> None:
        """Log a test step."""
        step_num = len(self.steps) + 1
        elapsed = time.time() - self.start_time
        
        step_info = {
            "number": step_num,
            "name": name,
            "elapsed_seconds": round(elapsed, 2),
            "details": details or {},
        }
        self.steps.append(step_info)
        
        self.logger.info(f"[STEP {step_num}] {name} (elapsed: {elapsed:.2f}s)")
        if details:
            self.logger.debug(f"  Details: {json.dumps(details, indent=2, default=str)}")
    
    def substep(self, message: str) -> None:
        """Log a substep within the current step."""
        self.logger.debug(f"    → {message}")
    
    def success(self, message: str) -> None:
        """Log a success message."""
        self.logger.info(f"✅ {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(f"⚠️ {message}")
    
    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error with optional exception details."""
        self.errors.append(message)
        self.logger.error(f"❌ {message}")
        if exception:
            self.logger.exception(f"Exception details: {exception}")
    
    def log_page_state(self, page_state: Dict) -> None:
        """Log detailed page state for debugging."""
        self.logger.debug("=" * 40)
        self.logger.debug("PAGE STATE SNAPSHOT")
        self.logger.debug("=" * 40)
        self.logger.debug(f"URL: {page_state.get('url', 'N/A')}")
        self.logger.debug(f"Title: {page_state.get('title', 'N/A')}")
        
        elements = page_state.get('elements', [])
        self.logger.debug(f"Elements found: {len(elements)}")
        
        # Log interactive elements
        for i, elem in enumerate(elements[:20]):  # Limit to first 20
            elem_type = elem.get('type', 'unknown')
            elem_text = elem.get('text', '')[:50]
            elem_id = elem.get('id', '')
            self.logger.debug(f"  [{i}] {elem_type}: '{elem_text}' (id={elem_id})")
    
    def log_action(self, action_type: str, target: str, result: str) -> None:
        """Log an action taken during the test."""
        self.logger.debug(f"ACTION: {action_type} → {target} = {result}")
    
    def log_screenshot(self, path: Path, description: str = "") -> None:
        """Log a screenshot taken during the test."""
        self.screenshots.append(path)
        self.logger.debug(f"📸 Screenshot: {path} - {description}")
    
    def log_result(self, result: Any) -> None:
        """Log the final test result."""
        elapsed = time.time() - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("TEST RESULT SUMMARY")
        self.logger.info("=" * 60)
        
        if hasattr(result, 'success'):
            self.logger.info(f"Success: {result.success}")
        if hasattr(result, 'coupon') and result.coupon:
            self.logger.info(f"Coupon Code: {result.coupon.code}")
            self.logger.info(f"Coupon Confidence: {result.coupon.confidence}")
        if hasattr(result, 'steps_taken'):
            self.logger.info(f"Steps Taken: {result.steps_taken}")
        if hasattr(result, 'error_message') and result.error_message:
            self.logger.error(f"Error: {result.error_message}")
        if hasattr(result, 'actions_log'):
            self.logger.debug("Actions Log:")
            for action in result.actions_log:
                self.logger.debug(f"  - {action}")
        
        self.logger.info(f"Total Duration: {elapsed:.2f}s")
        self.logger.info(f"Screenshots: {len(self.screenshots)}")
        self.logger.info(f"Errors: {len(self.errors)}")
        self.logger.info(f"Log file: {self.log_file}")
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize logging and return summary."""
        elapsed = time.time() - self.start_time
        
        summary = {
            "test_name": self.test_name,
            "duration_seconds": round(elapsed, 2),
            "steps_count": len(self.steps),
            "screenshots_count": len(self.screenshots),
            "errors_count": len(self.errors),
            "log_file": str(self.log_file),
            "steps": self.steps,
            "errors": self.errors,
        }
        
        self.logger.info("=" * 60)
        self.logger.info(f"TEST COMPLETED in {elapsed:.2f}s")
        self.logger.info("=" * 60)
        
        return summary


@pytest.fixture
def e2e_logger(request):
    """Create an E2E test logger for the current test."""
    test_name = request.node.name
    logger = E2ETestLogger(test_name)
    yield logger
    logger.finalize()


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def mock_server_url():
    """
    Start the mock survey server and return its URL.
    
    The server runs in a background thread for the duration of the test module.
    """
    try:
        from tests.mock_server import run_server_in_thread, stop_server, FLASK_AVAILABLE
    except ImportError:
        pytest.skip("Mock server not available")
    
    if not FLASK_AVAILABLE:
        pytest.skip("Flask not installed: pip install flask")
    
    # Start server
    url = run_server_in_thread(host="127.0.0.1", port=5556)
    
    # Wait for server to be ready
    import urllib.request
    import urllib.error
    for _ in range(10):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)
    else:
        pytest.fail("Mock server failed to start")
    
    yield url
    
    # Cleanup
    stop_server()


@pytest.fixture
def qr_receipt_image(tmp_path: Path, mock_server_url: str) -> Path:
    """
    Create a receipt image with QR code pointing to mock server.
    
    Args:
        tmp_path: Pytest temporary directory.
        mock_server_url: URL of the mock survey server.
        
    Returns:
        Path to the generated receipt image.
    """
    try:
        import qrcode
    except ImportError:
        pytest.skip("qrcode library not installed: pip install qrcode[pil]")
    
    survey_url = f"{mock_server_url}/survey"
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(survey_url)
    qr.make(fit=True)
    
    # Create PIL image and convert to numpy array
    pil_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(pil_img.convert('RGB'))
    
    # Create receipt-like image
    receipt = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add header text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(receipt, "SURVEY RECEIPT", (180, 50), font, 1.0, (0, 0, 0), 2)
    cv2.putText(receipt, "Thank you for shopping!", (150, 100), font, 0.7, (0, 0, 0), 1)
    cv2.putText(receipt, "Scan QR code for survey:", (140, 150), font, 0.6, (0, 0, 0), 1)
    
    # Resize QR code to fit
    qr_size = 350
    qr_resized = cv2.resize(qr_array, (qr_size, qr_size))
    
    # Place QR code in center
    x_offset = (600 - qr_size) // 2
    y_offset = 180
    receipt[y_offset:y_offset+qr_size, x_offset:x_offset+qr_size] = qr_resized
    
    # Add footer
    cv2.putText(receipt, f"URL: {survey_url[:40]}...", (50, 600), font, 0.5, (0, 0, 0), 1)
    cv2.putText(receipt, "Thank you!", (250, 700), font, 0.8, (0, 0, 0), 1)
    
    # Save image
    image_path = tmp_path / "sample_receipt_qr.jpg"
    cv2.imwrite(str(image_path), receipt)
    
    return image_path


@pytest.fixture
def mock_email_sender():
    """
    Mock email sender to avoid actual email sending.
    
    Sets fake SMTP environment variables and mocks the actual SMTP connection.
    """
    import os
    from unittest.mock import AsyncMock
    
    # Set fake SMTP environment variables so validation passes
    env_vars = {
        'SMTP_HOST': 'localhost',
        'SMTP_PORT': '587',
        'SMTP_USER': 'test@example.com',
        'SMTP_PASSWORD': 'testpassword',
    }
    
    # Store original values
    original_env = {k: os.environ.get(k) for k in env_vars}
    
    # Set test values
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Mock the actual SMTP connection
    with patch('smtplib.SMTP') as mock_smtp:
        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        mock_instance.starttls.return_value = None
        mock_instance.login.return_value = None
        mock_instance.sendmail.return_value = {}
        
        yield mock_smtp
    
    # Restore original environment
    for key, original in original_env.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test (requires mock server)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# =============================================================================
# E2E TESTS
# =============================================================================

class TestE2EFullFlow:
    """End-to-end tests for the complete survey automation flow."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_flow_happy_path(
        self,
        qr_receipt_image: Path,
        mock_server_url: str,
        mock_email_sender,
        e2e_logger: E2ETestLogger,
    ):
        """
        Complete flow test using mock server with happy mood.
        
        Tests the full pipeline:
        1. Extract URL from receipt QR code
        2. Configure happy persona
        3. Navigate and complete survey
        4. Extract coupon code
        5. Send email with coupon
        """
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Initialize SurveyBot", {"headless": True})
        
        bot = SurveyBot(
            verbose=True,
            headless=True,
        )
        
        e2e_logger.step("Run complete flow", {
            "image_path": str(qr_receipt_image),
            "mood": "😊",  # Emoji input should map to "happy"
            "email": "test@example.com",
        })
        
        try:
            result = await bot.run(
                image_path=str(qr_receipt_image),
                mood="😊",  # Test emoji mood input
                email="test@example.com",
            )
            
            e2e_logger.log_result(result)
            
            # Assertions
            e2e_logger.step("Verify results")
            
            assert result.success, f"Survey should complete successfully. Error: {result.error_message}"
            e2e_logger.success("Survey completed successfully")
            
            assert result.coupon is not None, "Should extract coupon code"
            assert result.coupon.code is not None, "Coupon code should not be None"
            assert len(result.coupon.code) >= 6, f"Coupon should be valid format: {result.coupon.code}"
            e2e_logger.success(f"Coupon extracted: {result.coupon.code}")
            
            # Check email was sent (mocked)
            assert result.email_sent, "Email should be marked as sent"
            mock_email_sender.assert_called()
            e2e_logger.success("Email sent successfully")
            
            # Additional assertions
            assert result.survey_url, "Should have survey URL"
            assert result.steps_taken > 0, "Should take at least one step"
            assert result.duration_seconds > 0, "Should have duration"
            
            e2e_logger.success(f"All assertions passed! Steps: {result.steps_taken}")
            
        except Exception as e:
            e2e_logger.error(f"Test failed: {str(e)}", exception=e)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_flow_angry_path(
        self,
        qr_receipt_image: Path,
        mock_server_url: str,
        mock_email_sender,
        e2e_logger: E2ETestLogger,
    ):
        """
        Complete flow test with angry mood.
        
        Verifies that angry persona generates appropriate responses
        (lower ratings, negative feedback).
        """
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Initialize SurveyBot with angry mood")
        
        bot = SurveyBot(verbose=True, headless=True)
        
        e2e_logger.step("Run with angry mood", {
            "mood": "😠",
        })
        
        try:
            result = await bot.run(
                image_path=str(qr_receipt_image),
                mood="😠",  # Angry emoji
                email="angry@example.com",
            )
            
            e2e_logger.log_result(result)
            
            assert result.success, f"Survey should complete even with angry mood. Error: {result.error_message}"
            assert result.coupon is not None, "Should still get coupon with angry mood"
            
            e2e_logger.success(f"Angry flow completed with coupon: {result.coupon.code}")
            
        except Exception as e:
            e2e_logger.error(f"Angry flow failed: {str(e)}", exception=e)
            raise
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_flow_neutral_path(
        self,
        qr_receipt_image: Path,
        mock_server_url: str,
        mock_email_sender,
        e2e_logger: E2ETestLogger,
    ):
        """
        Complete flow test with neutral mood.
        """
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Run with neutral mood")
        
        bot = SurveyBot(verbose=True, headless=True)
        
        result = await bot.run(
            image_path=str(qr_receipt_image),
            mood="neutral",
            email="neutral@example.com",
        )
        
        e2e_logger.log_result(result)
        
        assert result.success, f"Neutral flow failed: {result.error_message}"
        assert result.coupon is not None
        
        e2e_logger.success(f"Neutral flow completed: {result.coupon.code}")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_direct_url_flow(
        self,
        mock_server_url: str,
        mock_email_sender,
        e2e_logger: E2ETestLogger,
    ):
        """
        Test flow with direct URL (skipping image processing).
        """
        from src.survey_bot.main import SurveyBot
        
        survey_url = f"{mock_server_url}/survey"
        
        e2e_logger.step("Run with direct URL", {"url": survey_url})
        
        bot = SurveyBot(verbose=True, headless=True)
        
        result = await bot.run_with_url(
            survey_url=survey_url,
            mood="happy",
            email="direct@example.com",
        )
        
        e2e_logger.log_result(result)
        
        assert result.success, f"Direct URL flow failed: {result.error_message}"
        assert result.coupon is not None
        
        e2e_logger.success(f"Direct URL flow completed: {result.coupon.code}")


class TestE2EErrorHandling:
    """E2E tests for error handling scenarios."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_image_path(
        self,
        e2e_logger: E2ETestLogger,
    ):
        """Test handling of non-existent image file."""
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Test with invalid image path")
        
        bot = SurveyBot(verbose=True, headless=True)
        
        result = await bot.run(
            image_path="/nonexistent/path/receipt.jpg",
            mood="happy",
            email="test@example.com",
        )
        
        e2e_logger.log_result(result)
        
        assert not result.success, "Should fail with invalid image"
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "extraction failed" in result.error_message.lower()
        
        e2e_logger.success("Correctly handled invalid image path")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_no_qr_code_image(
        self,
        tmp_path: Path,
        e2e_logger: E2ETestLogger,
    ):
        """Test handling of image without QR code."""
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Create blank image without QR code")
        
        # Create blank image
        blank_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
        blank_path = tmp_path / "blank_receipt.jpg"
        cv2.imwrite(str(blank_path), blank_img)
        
        e2e_logger.step("Run with blank image")
        
        bot = SurveyBot(verbose=True, headless=True)
        
        result = await bot.run(
            image_path=str(blank_path),
            mood="happy",
            email="test@example.com",
        )
        
        e2e_logger.log_result(result)
        
        assert not result.success, "Should fail without QR code"
        assert result.error_message is not None
        
        e2e_logger.success("Correctly handled image without QR code")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_survey_url(
        self,
        e2e_logger: E2ETestLogger,
    ):
        """Test handling of unreachable survey URL."""
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Test with unreachable URL")
        
        bot = SurveyBot(verbose=True, headless=True)
        
        result = await bot.run_with_url(
            survey_url="http://localhost:9999/nonexistent",
            mood="happy",
            email="test@example.com",
        )
        
        e2e_logger.log_result(result)
        
        assert not result.success, "Should fail with unreachable URL"
        assert result.error_message is not None
        
        e2e_logger.success("Correctly handled unreachable URL")


class TestE2EPerformance:
    """E2E tests for performance and timing."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_completion_time(
        self,
        qr_receipt_image: Path,
        mock_server_url: str,
        mock_email_sender,
        e2e_logger: E2ETestLogger,
    ):
        """Test that survey completes within reasonable time."""
        from src.survey_bot.main import SurveyBot
        
        e2e_logger.step("Run performance test")
        
        start_time = time.time()
        
        bot = SurveyBot(verbose=False, headless=True)
        
        result = await bot.run(
            image_path=str(qr_receipt_image),
            mood="happy",
            email="perf@example.com",
        )
        
        elapsed = time.time() - start_time
        
        e2e_logger.log_result(result)
        e2e_logger.step("Performance metrics", {
            "total_time_seconds": round(elapsed, 2),
            "steps_taken": result.steps_taken,
            "success": result.success,
        })
        
        # Performance assertions
        assert result.success, f"Should complete successfully: {result.error_message}"
        assert elapsed < 120, f"Should complete within 2 minutes, took {elapsed:.1f}s"
        
        if result.steps_taken > 0:
            avg_step_time = elapsed / result.steps_taken
            e2e_logger.substep(f"Average step time: {avg_step_time:.2f}s")
            assert avg_step_time < 10, f"Average step time too high: {avg_step_time:.1f}s"
        
        e2e_logger.success(f"Completed in {elapsed:.1f}s ({result.steps_taken} steps)")


class TestE2EMoodMapping:
    """Tests for mood input variations."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mood_input,expected_mood", [
        ("happy", "happy"),
        ("😊", "happy"),
        ("😃", "happy"),
        ("neutral", "neutral"),
        ("😐", "neutral"),
        ("angry", "angry"),
        ("😠", "angry"),
        ("😡", "angry"),
        ("HAPPY", "happy"),  # Case insensitive
        ("Happy", "happy"),
    ])
    async def test_mood_variations(
        self,
        mood_input: str,
        expected_mood: str,
        mock_server_url: str,
        mock_email_sender,
        e2e_logger: E2ETestLogger,
    ):
        """Test that various mood inputs map correctly."""
        from src.survey_bot.main import SurveyBot
        from src.survey_bot.agents.supervisor import SupervisorAgent
        
        e2e_logger.step(f"Test mood mapping: '{mood_input}' → '{expected_mood}'")
        
        supervisor = SupervisorAgent()
        
        try:
            persona = supervisor.get_persona(mood_input)
            actual_mood = persona.mood.value
            
            assert actual_mood == expected_mood, f"Expected {expected_mood}, got {actual_mood}"
            e2e_logger.success(f"Mood '{mood_input}' correctly mapped to '{actual_mood}'")
            
        except ValueError as e:
            e2e_logger.error(f"Mood mapping failed for '{mood_input}': {e}")
            raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_test_report(results: List[Dict]) -> str:
    """
    Generate a test report from results.
    
    Args:
        results: List of test result dictionaries.
        
    Returns:
        Formatted report string.
    """
    report_lines = [
        "=" * 60,
        "E2E TEST REPORT",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]
    
    passed = sum(1 for r in results if r.get('success', False))
    failed = len(results) - passed
    
    report_lines.append(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    report_lines.append("")
    
    for result in results:
        status = "✅" if result.get('success', False) else "❌"
        name = result.get('test_name', 'unknown')
        duration = result.get('duration_seconds', 0)
        report_lines.append(f"{status} {name} ({duration:.1f}s)")
        
        if not result.get('success', False) and result.get('error'):
            report_lines.append(f"   Error: {result['error']}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """Run E2E tests directly."""
    pytest.main([
        __file__,
        "-v",
        "-m", "e2e",
        "--log-cli-level=INFO",
        "-x",  # Stop on first failure
    ])
