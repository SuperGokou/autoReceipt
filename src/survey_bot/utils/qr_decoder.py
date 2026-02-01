from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol

if TYPE_CHECKING:
    from cv2.typing import MatLike
else:
    MatLike = np.ndarray

__all__ = ["extract_qr_urls", "preprocess_for_qr"]

logger = logging.getLogger(__name__)


def preprocess_for_qr(image: MatLike) -> MatLike:
    """
    Preprocess an image to improve QR code detection.

    Applies grayscale conversion, noise reduction, and adaptive
    thresholding to enhance QR code visibility.

    Args:
        image: Input image as numpy array (BGR or grayscale).

    Returns:
        Preprocessed grayscale image optimized for QR detection.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding for varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


def _decode_qr_from_image(image: MatLike) -> list[str]:
    """Decode QR codes and return their string contents."""
    try:
        results = decode(image, symbols=[ZBarSymbol.QRCODE])
        return [r.data.decode("utf-8") for r in results]
    except Exception as e:
        logger.debug(f"QR decode failed: {e}")
        return []


def extract_qr_urls(image_source: str | Path | MatLike) -> list[str]:
    """
    Extract URLs from QR codes in an image.

    Attempts multiple strategies to find QR codes:
    1. Direct decode of original image
    2. Decode after preprocessing
    3. Try 4 rotations (0°, 90°, 180°, 270°)

    Args:
        image_source: File path or numpy array of the image.

    Returns:
        List of unique URLs found in QR codes.

    Raises:
        ValueError: If image cannot be loaded or is invalid.

    Example:
        >>> urls = extract_qr_urls("receipt.jpg")
        >>> if urls:
        ...     print(f"Found survey URL: {urls[0]}")
    """
    # Load image if path provided
    image: MatLike
    if isinstance(image_source, (str, Path)):
        loaded_image = cv2.imread(str(image_source))
        if loaded_image is None:
            raise ValueError(f"Could not load image: {image_source}")
        image = loaded_image
    else:
        image = image_source

    if image is not None and hasattr(image, 'size') and image.size == 0:
        raise ValueError("Image is empty or corrupt")

    found_urls: set[str] = set()

    # Strategy 1: Try original image
    decoded = _decode_qr_from_image(image)
    found_urls.update(decoded)

    if found_urls:
        logger.debug("QR found in original image")
        return _filter_urls(found_urls)

    # Strategy 2: Try preprocessed image
    preprocessed = preprocess_for_qr(image)
    decoded = _decode_qr_from_image(preprocessed)
    found_urls.update(decoded)

    if found_urls:
        logger.debug("QR found after preprocessing")
        return _filter_urls(found_urls)

    # Strategy 3: Try rotations
    rotations = [
        (cv2.ROTATE_90_CLOCKWISE, "90°"),
        (cv2.ROTATE_180, "180°"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "270°"),
    ]

    for rotation, label in rotations:
        # Try rotated original
        rotated = cv2.rotate(image, rotation)
        decoded = _decode_qr_from_image(rotated)
        found_urls.update(decoded)

        # Try rotated preprocessed
        rotated_prep = cv2.rotate(preprocessed, rotation)
        decoded = _decode_qr_from_image(rotated_prep)
        found_urls.update(decoded)

        if found_urls:
            logger.debug(f"QR found at {label} rotation")
            break

    return _filter_urls(found_urls)


def _filter_urls(strings: set[str]) -> list[str]:
    """Filter strings to only include valid URLs."""
    return [
        s for s in strings
        if s.startswith(("http://", "https://"))
    ]