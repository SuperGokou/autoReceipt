from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

__all__ = [
    "preprocess_receipt_image",
    "extract_text_ocr",
    "find_survey_url",
]

logger = logging.getLogger(__name__)

# Known survey domains for priority matching
SURVEY_DOMAINS = [
    r"medallia\.com",
    r"smg\.com",
    r"survey\.walmart\.com",
    r"tell\w+\.com",  # tellpopeyes.com, telldunkin.com, etc.
    r"mcdvoice\.com",
    r"mydunkin\.com",
    r"myopinion\.\w+",
    r"\w*survey\w*\.com",  # Any domain with 'survey'
]


pytesseract.pytesseract.tesseract_cmd = r'D:\Study\Tesseract-OCR\tesseract.exe'

def _detect_skew_angle(image: np.ndarray) -> float:
    """
    Detect the skew angle of text in an image.

    Uses contour detection and minAreaRect to find
    the dominant text angle.

    Args:
        image: Grayscale image.

    Returns:
        Skew angle in degrees (-45 to 45 range).
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    # Get angles from rotated rectangles
    angles = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Skip tiny contours
            continue
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]

        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        angles.append(angle)

    if not angles:
        return 0.0

    # Return median angle (robust to outliers)
    return float(np.median(angles))


def _clean_url(url: str) -> str:
    """
    Clean OCR artifacts from extracted URL.

    Fixes common OCR errors in URLs:
    - Removes accidental whitespace
    - Fixes 'corn' → 'com' typo
    - Handles l/1 and O/0 confusion
    """
    # Remove all whitespace
    url = re.sub(r'\s+', '', url)

    # Common OCR fixes
    url = re.sub(r'\.corn\b', '.com', url, flags=re.IGNORECASE)
    url = re.sub(r'\.corm\b', '.com', url, flags=re.IGNORECASE)

    # Ensure protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    return url


def preprocess_receipt_image(image_path: str | Path) -> np.ndarray:
    """
    Preprocess a receipt image for optimal OCR results.

    Applies multiple image processing steps to improve text
    recognition accuracy on receipt-style documents.

    Args:
        image_path: Path to the receipt image file.

    Returns:
        Preprocessed image as numpy array ready for OCR.

    Raises:
        FileNotFoundError: If image path doesn't exist.
        ValueError: If image cannot be loaded.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")

    logger.debug(f"Loaded image: {image.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect and correct skew
    angle = _detect_skew_angle(gray)
    if abs(angle) > 0.5:  # Only rotate if significant
        logger.debug(f"Deskewing by {angle:.1f} degrees")
        h, w = gray.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(
            gray, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Increase contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    # Binary threshold for cleaner text
    _, binary = cv2.threshold(
        contrasted, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    logger.debug("Preprocessing complete")
    return binary


def extract_text_ocr(image: np.ndarray) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Configured for receipt-style text with sparse layout
    and mixed content types.

    Args:
        image: Preprocessed image as numpy array.

    Returns:
        Raw extracted text as string (empty string on failure).
    """
    # Tesseract config for sparse receipt text
    custom_config = r'--oem 3 --psm 11'

    try:
        text = pytesseract.image_to_string(
            image,
            lang='eng',
            config=custom_config
        )
        logger.debug(f"OCR extracted {len(text)} characters")
        return text

    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract not found! Install with: brew install tesseract"
        )
        return ""
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def find_survey_url(text: str) -> str | None:
    """
    Extract survey URL from OCR text using regex patterns.

    Searches for URLs matching common survey providers with
    fallback to generic URL detection.

    Args:
        text: Raw OCR text to search.

    Returns:
        First valid survey URL found, or None if not found.

    Example:
        >>> text = "Visit survey.walmart.com/r/abc123 for a chance to win!"
        >>> find_survey_url(text)
        'https://survey.walmart.com/r/abc123'
    """
    if not text:
        return None

    # Normalize text
    text = text.replace('\n', ' ')

    # Strategy 1: Look for known survey domains first
    for domain_pattern in SURVEY_DOMAINS:
        pattern = rf'(https?://)?({domain_pattern}[/\w\-\.\?\=\&]*)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group(0)
            logger.debug(f"Found survey domain URL: {url}")
            return _clean_url(url)

    # Strategy 2: Any URL containing 'survey' keyword
    survey_url_pattern = r'(https?://)?[\w\-\.]*survey[\w\-\.]*\.[a-z]{2,}[/\w\-\.\?\=\&]*'
    match = re.search(survey_url_pattern, text, re.IGNORECASE)
    if match:
        url = match.group(0)
        logger.debug(f"Found survey keyword URL: {url}")
        return _clean_url(url)

    # Strategy 3: Generic URL as last resort
    generic_pattern = r'https?://[\w\-\.]+\.[a-z]{2,}[/\w\-\.\?\=\&]*'
    match = re.search(generic_pattern, text, re.IGNORECASE)
    if match:
        url = match.group(0)
        logger.debug(f"Found generic URL: {url}")
        return _clean_url(url)

    logger.debug("No URL found in text")
    return None



