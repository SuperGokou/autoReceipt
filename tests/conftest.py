"""
Pytest configuration and fixtures for survey_bot tests.

This module provides reusable test fixtures including:
- Generated test images (QR codes, text receipts, blank images)
- Temporary file handling
- Common test utilities
- Integration test fixtures (browser, observer, interactor)

Fixtures are auto-generated at test runtime using tmp_path,
so no manual image creation is needed.
"""
from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np
import pytest


# =============================================================================
# IMAGE FIXTURES (Module 1 - Ingestion)
# =============================================================================


@pytest.fixture
def qr_code_image(tmp_path: Path) -> Path:
    """
    Create a test image with a QR code containing a survey URL.
    
    The QR code encodes: https://survey.example.com/test123
    
    Args:
        tmp_path: Pytest's temporary directory fixture.
        
    Returns:
        Path to the generated QR code image.
        
    Note:
        Requires qrcode library: pip install qrcode[pil]
    """
    try:
        import qrcode
    except ImportError:
        pytest.skip("qrcode library not installed: pip install qrcode[pil]")
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data("https://survey.example.com/test123")
    qr.make(fit=True)
    
    # Create PIL image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save to temp path
    image_path = tmp_path / "qr_receipt.png"
    img.save(str(image_path))
    
    return image_path


@pytest.fixture
def text_url_image(tmp_path: Path) -> Path:
    """
    Create a test image with a text URL (no QR code).
    
    Simulates a receipt with printed survey URL that will
    be detected by OCR.
    
    Args:
        tmp_path: Pytest's temporary directory fixture.
        
    Returns:
        Path to the generated text receipt image.
    """
    # Create white background (800x600, 3 channels for BGR)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = (0, 0, 0)
    
    # Store header
    cv2.putText(
        img, "EXAMPLE STORE",
        (250, 80),
        font, 1.2, black, 2, cv2.LINE_AA
    )
    
    # Separator line
    cv2.putText(
        img, "=" * 40,
        (150, 130),
        font, 0.6, black, 1, cv2.LINE_AA
    )
    
    # Receipt content
    cv2.putText(
        img, "Thank you for shopping!",
        (220, 200),
        font, 0.7, black, 1, cv2.LINE_AA
    )
    
    cv2.putText(
        img, "Date: 2024-01-15",
        (280, 250),
        font, 0.6, black, 1, cv2.LINE_AA
    )
    
    # Survey invitation (key text for OCR)
    cv2.putText(
        img, "Take our survey at:",
        (230, 330),
        font, 0.7, black, 1, cv2.LINE_AA
    )
    
    # The URL - this is what OCR needs to find
    cv2.putText(
        img, "https://survey.teststore.com/abc",
        (140, 380),
        font, 0.65, black, 2, cv2.LINE_AA
    )
    
    # Footer
    cv2.putText(
        img, "=" * 40,
        (150, 450),
        font, 0.6, black, 1, cv2.LINE_AA
    )
    
    cv2.putText(
        img, "Have a great day!",
        (260, 520),
        font, 0.7, black, 1, cv2.LINE_AA
    )
    
    # Save image
    image_path = tmp_path / "text_receipt.png"
    cv2.imwrite(str(image_path), img)
    
    return image_path


@pytest.fixture
def blank_image(tmp_path: Path) -> Path:
    """
    Create a blank gray image for failure testing.
    
    This image contains no QR codes or text, so extraction
    should fail gracefully.
    
    Args:
        tmp_path: Pytest's temporary directory fixture.
        
    Returns:
        Path to the generated blank image.
    """
    # Solid gray image (no text, no QR)
    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    
    image_path = tmp_path / "blank.png"
    cv2.imwrite(str(image_path), img)
    
    return image_path


@pytest.fixture
def corrupted_file(tmp_path: Path) -> Path:
    """
    Create a corrupted/invalid image file.
    
    This file has a .png extension but contains invalid data,
    testing error handling for corrupt files.
    
    Args:
        tmp_path: Pytest's temporary directory fixture.
        
    Returns:
        Path to the corrupted file.
    """
    file_path = tmp_path / "corrupted.png"
    
    # Write garbage data (not a valid PNG)
    file_path.write_bytes(b"This is definitely not a valid PNG file! \x00\x01\x02\x03")
    
    return file_path


@pytest.fixture
def sample_receipt_text() -> str:
    """
    Sample OCR text output for URL detection testing.
    
    This fixture provides pre-extracted text to test the
    find_survey_url function without needing OCR.
    
    Returns:
        Sample receipt text containing a survey URL.
    """
    return """
    WALMART SUPERCENTER
    Store #1234
    123 Main Street
    Anytown, USA 12345
    
    ========================
    
    GROCERIES
    MILK 2%           $3.99
    BREAD             $2.49
    EGGS              $4.99
    
    SUBTOTAL         $11.47
    TAX               $0.92
    TOTAL            $12.39
    
    ========================
    
    Thank you for shopping!
    
    Tell us about your visit at
    survey.walmart.com/r/ABC123XYZ
    
    for a chance to win a
    $1000 gift card!
    
    ========================
    
    Transaction: 1234567890
    Date: 01/15/2024 14:32
    """


@pytest.fixture
def high_contrast_text_image(tmp_path: Path) -> Path:
    """
    Create a high-contrast text image for reliable OCR testing.
    
    Uses larger font and better spacing for more reliable
    OCR extraction.
    
    Args:
        tmp_path: Pytest's temporary directory fixture.
        
    Returns:
        Path to the generated image.
    """
    # Larger canvas for bigger text
    img = np.ones((400, 1000, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = (0, 0, 0)
    
    # Large, clear URL text
    cv2.putText(
        img, "https://survey.example.com/test",
        (50, 200),
        font, 1.5, black, 3, cv2.LINE_AA
    )
    
    image_path = tmp_path / "high_contrast.png"
    cv2.imwrite(str(image_path), img)
    
    return image_path


# =============================================================================
# INTEGRATION TEST FIXTURES (Module 3 - Navigation)
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def supervisor_agent():
    """Create a SupervisorAgent instance."""
    from src.survey_bot.agents.supervisor import SupervisorAgent
    return SupervisorAgent()


@pytest.fixture
def page_observer():
    """Create a PageObserver instance for tests."""
    from src.survey_bot.browser.observer import PageObserver
    return PageObserver()


@pytest.fixture
def page_interactor():
    """Create a PageInteractor with fast settings for tests."""
    from src.survey_bot.browser.interactor import PageInteractor
    return PageInteractor(
        human_like=False,  # Disable delays for faster tests
        max_retries=2,
    )


@pytest.fixture
def headless_browser_manager():
    """Create a headless BrowserManager for tests."""
    from src.survey_bot.browser.launcher import BrowserManager
    return BrowserManager(headless=True)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require server)"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test (requires mock server)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration and e2e tests unless explicitly requested."""
    # Get marker options
    marker_option = config.getoption("-m", default="")
    
    # Check if running integration tests
    run_integration = (
        "integration" in marker_option or
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true"
    )
    
    # Check if running e2e tests
    run_e2e = (
        "e2e" in marker_option or
        os.environ.get("RUN_E2E_TESTS", "").lower() == "true"
    )
    
    # Skip markers
    skip_integration = pytest.mark.skip(
        reason="Integration tests skipped by default. Use -m integration or set RUN_INTEGRATION_TESTS=true"
    )
    skip_e2e = pytest.mark.skip(
        reason="E2E tests skipped by default. Use -m e2e or set RUN_E2E_TESTS=true"
    )
    
    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)
