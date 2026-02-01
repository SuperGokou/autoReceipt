"""
Ingestion Agent for receipt image processing.

This module provides the IngestionAgent class which handles:
- QR code detection and decoding
- OCR text extraction for URLs
- Vision LLM extraction (Qwen VL, LLaVA, etc.)

The agent is the first step in the survey automation pipeline,
responsible for extracting survey URLs from receipt images.

Example Usage:
    >>> from pathlib import Path
    >>> from survey_bot.agents.ingestion import IngestionAgent
    >>>
    >>> agent = IngestionAgent()
    >>> result = await agent.extract_from_image("receipt.jpg")
    >>>
    >>> if result.success:
    ...     print(f"Found URL: {result.data.url}")
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Union

from ..models.receipt import ExtractionResult, ReceiptData

__all__ = ["IngestionAgent"]

logger = logging.getLogger(__name__)

# URL patterns for survey detection
SURVEY_URL_PATTERNS = [
    # Direct survey URLs
    r'https?://[^\s<>"{}|\\^`\[\]]+survey[^\s<>"{}|\\^`\[\]]*',
    r'https?://survey\.[^\s<>"{}|\\^`\[\]]+',
    # Common survey domains
    r'https?://(?:www\.)?pandaexpress\.com/feedback[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?pandaguestexperience\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?tellwm\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?mcdvoice\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?talktowendys\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?myopinion\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?tellthebell\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?mybkexperience\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?tellsubway\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?mycfavisit\.com[^\s<>"{}|\\^`\[\]]*',
    r'https?://(?:www\.)?mystarbucksvisit\.com[^\s<>"{}|\\^`\[\]]*',
    # Domain-only patterns (no https prefix in receipt)
    r'pandaexpress\.com/feedback',
    r'pandaguestexperience\.com',
    r'survey\.walmart\.com[^\s<>"{}|\\^`\[\]]*',
    # Generic URL pattern (fallback)
    r'https?://[a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,}[^\s<>"{}|\\^`\[\]]*',
]

# Known survey domain patterns (without protocol)
KNOWN_SURVEY_DOMAINS = [
    "pandaexpress.com/feedback",
    "pandaguestexperience.com",
    "survey.walmart.com",
    "tellwm.com",
    "mcdvoice.com",
    "talktowendys.com",
    "tellthebell.com",
    "mybkexperience.com",
    "tellsubway.com",
    "mystarbucksvisit.com",
    "mycfavisit.com",
    "talktosonic.com",
    "tellpizzahut.com",
    "tellpopeyes.com",
    "whataburgersurvey.com",
    "jacklistens.com",
]

# Vision LLM prompt for extracting survey info
VISION_PROMPT = """Look at this receipt image and extract the following information:

1. Survey URL - Look for any website URL related to surveys, feedback, or customer satisfaction (e.g., pandaexpress.com/feedback, survey.walmart.com, etc.)
2. Survey Code - Look for any validation code, survey code, or reference number (usually a long number like 2125-7062-1388-0079-0312-9406)
3. Store Name - The name of the business/restaurant

Respond in this exact format:
URL: [the survey url or "not found"]
CODE: [the survey code or "not found"]
STORE: [the store name or "not found"]

Only extract what you can clearly see. Do not make up information."""


class IngestionAgent:
    """
    Agent for extracting survey URLs from receipt images.

    Handles the first stage of survey automation:
    1. Try Vision LLM (Qwen VL, LLaVA) if configured - most accurate
    2. Try QR code detection (fast, reliable for QR codes)
    3. Fall back to OCR for text URLs

    Attributes:
        QR_CONFIDENCE: Confidence score for QR extractions.
        OCR_CONFIDENCE: Confidence score for OCR extractions.
        VISION_CONFIDENCE: Confidence score for Vision LLM extractions.
    """

    # Confidence scores for different extraction methods
    QR_CONFIDENCE = 0.95
    OCR_CONFIDENCE = 0.75
    VISION_CONFIDENCE = 0.90

    def __init__(
            self,
            vision_api_key: Optional[str] = None,
            use_vision_llm: bool = True,
            ollama_model: str = "qwen3-vl",
            ollama_host: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the IngestionAgent.

        Args:
            vision_api_key: Optional API key for cloud vision services.
            use_vision_llm: Whether to use local Vision LLM (Ollama).
            ollama_model: Ollama model name (default: qwen2.5-vl).
            ollama_host: Ollama API host (default: http://localhost:11434).
        """
        self.vision_api_key = vision_api_key
        self.use_vision_llm = use_vision_llm
        self.ollama_model = os.environ.get("OLLAMA_VISION_MODEL", ollama_model)
        from ..llm.ollama_host import detect_ollama_host
        self.ollama_host = ollama_host or detect_ollama_host()
        logger.debug(f"IngestionAgent initialized (vision_llm={use_vision_llm}, model={self.ollama_model})")

    async def extract_from_image(
            self,
            image_path: Union[str, Path],
    ) -> ExtractionResult:
        """
        Extract survey URL from a receipt image.

        Tries multiple extraction methods in order of reliability:
        1. Vision LLM (Qwen VL via Ollama) - best accuracy
        2. QR code detection - fast for QR codes
        3. OCR text extraction - fallback

        Args:
            image_path: Path to the receipt image.

        Returns:
            ExtractionResult with success status and extracted data.
        """
        start_time = time.time()
        image_path = Path(image_path)

        # Validate file exists
        if not image_path.exists():
            return ExtractionResult(
                success=False,
                error=f"File not found: {image_path}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Try Vision LLM first (most accurate for receipts)
        if self.use_vision_llm:
            try:
                result = await self._extract_vision_llm(image_path)
                if result:
                    logger.info(f"Vision LLM extracted URL: {result.survey_url}")
                    return ExtractionResult(
                        success=True,
                        data=result,
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
            except Exception as e:
                logger.warning(f"Vision LLM extraction failed: {e}")

        # Try QR code detection
        try:
            result = await self._extract_qr(image_path)
            if result:
                logger.info(f"QR extracted URL: {result.survey_url}")
                return ExtractionResult(
                    success=True,
                    data=result,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
        except Exception as e:
            logger.debug(f"QR extraction failed: {e}")

        # Try OCR as fallback
        try:
            result = await self._extract_ocr(image_path)
            if result:
                logger.info(f"OCR extracted URL: {result.survey_url}")
                return ExtractionResult(
                    success=True,
                    data=result,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")

        # Try Cloud Vision API if available
        if self.vision_api_key:
            try:
                result = await self._extract_vision(image_path)
                if result:
                    return ExtractionResult(
                        success=True,
                        data=result,
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
            except Exception as e:
                logger.debug(f"Vision API extraction failed: {e}")

        # No URL found
        return ExtractionResult(
            success=False,
            error="No survey URL found in image",
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    async def _extract_qr(self, image_path: Path) -> Optional[ReceiptData]:
        """
        Extract URL from QR code in image.

        Args:
            image_path: Path to the image file.

        Returns:
            ReceiptData if QR found, None otherwise.
        """
        try:
            import cv2
            from pyzbar import pyzbar
        except ImportError:
            logger.warning("pyzbar not installed: pip install pyzbar")
            return None

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Detect QR codes
        decoded = pyzbar.decode(img)
        logger.info(f"QR detection found {len(decoded)} codes")

        for obj in decoded:
            logger.info(f"Found code type: {obj.type}, data: {obj.data.decode('utf-8')[:100]}")
            if obj.type == "QRCODE":
                data = obj.data.decode("utf-8")

                # Check if it looks like a URL
                if self._is_survey_url(data):
                    # Add protocol if missing
                    url = data
                    if not url.startswith(("http://", "https://")):
                        url = f"https://www.{url}"

                    # Extract survey code from URL if present (e.g., ?cn=XXXX)
                    survey_code = None
                    import urllib.parse
                    try:
                        parsed = urllib.parse.urlparse(url)
                        params = urllib.parse.parse_qs(parsed.query)
                        if "cn" in params:
                            survey_code = params["cn"][0]
                            logger.info(f"QR extracted survey code: {survey_code}")
                    except Exception as e:
                        logger.debug(f"Could not parse URL params: {e}")

                    return ReceiptData(
                        url=url,
                        survey_url=url,
                        survey_code=survey_code,
                        extraction_method="qr",
                        confidence=self.QR_CONFIDENCE,
                    )
                else:
                    logger.info(f"QR data is not a survey URL: {data[:100]}")

        return None

    async def _extract_vision_llm(self, image_path: Path) -> Optional[ReceiptData]:
        """
        Extract URL using Vision LLM (Qwen VL via Ollama).

        Args:
            image_path: Path to the image file.

        Returns:
            ReceiptData if URL found, None otherwise.
        """
        import httpx

        # Read and encode image as base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Prepare Ollama API request
        payload = {
            "model": self.ollama_model,
            "prompt": VISION_PROMPT,
            "images": [image_data],
            "stream": False,
        }

        logger.info(f"Calling Ollama Vision LLM ({self.ollama_model})...")

        # Use longer timeout for vision model
        timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,  # Vision models can be slow
            write=30.0,
            pool=10.0,
        )

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json()
                    break  # Success, exit retry loop

            except httpx.ConnectError as e:
                last_error = f"Cannot connect to Ollama at {self.ollama_host}. Is Ollama running? Run: ollama serve"
                logger.error(last_error)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise Exception(last_error)

            except httpx.TimeoutException as e:
                last_error = f"Ollama request timed out (attempt {attempt + 1}). Model may be loading..."
                logger.warning(last_error)
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                else:
                    raise Exception(last_error)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise Exception(f"Model '{self.ollama_model}' not found. Run: ollama pull {self.ollama_model}")
                logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
                raise

        # Parse response
        response_text = result.get("response", "")
        logger.info(f"Vision LLM response:\n{response_text}")

        # Extract URL from response
        url = None
        survey_code = None
        store_name = None

        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("URL:"):
                value = line[4:].strip()
                if value.lower() != "not found" and value != "-":
                    url = value
                    # Clean up URL
                    url = url.strip('"\'<>')
                    # Add protocol if missing
                    if url and not url.startswith(("http://", "https://")):
                        url = f"https://{url}"
            elif line.upper().startswith("CODE:"):
                value = line[5:].strip()
                if value.lower() != "not found" and value != "-":
                    survey_code = value.strip('"\'')
            elif line.upper().startswith("STORE:"):
                value = line[6:].strip()
                if value.lower() != "not found" and value != "-":
                    store_name = value.strip('"\'')

        if url:
            return ReceiptData(
                url=url,
                survey_url=url,
                extraction_method="vision_llm",
                confidence=self.VISION_CONFIDENCE,
                store_name=store_name,
                survey_code=survey_code,
            )

        # If no URL but we have a known store, try to construct URL
        if store_name:
            store_lower = store_name.lower()
            if "panda" in store_lower:
                url = "https://www.pandaexpress.com/feedback"
                return ReceiptData(
                    url=url,
                    survey_url=url,
                    extraction_method="vision_llm",
                    confidence=self.VISION_CONFIDENCE * 0.9,
                    store_name=store_name,
                    survey_code=survey_code,
                )

        return None

    async def _extract_ocr(self, image_path: Path) -> Optional[ReceiptData]:
        """
        Extract URL using OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            ReceiptData if URL found, None otherwise.
        """
        try:
            import pytesseract
            import cv2
        except ImportError:
            logger.warning("pytesseract not installed: pip install pytesseract")
            return None

        # Read and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold for better OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        try:
            text = pytesseract.image_to_string(thresh)
            logger.info(f"OCR extracted {len(text)} characters")
            logger.debug(f"OCR text sample: {text[:500]}")
        except pytesseract.TesseractError as e:
            error_msg = str(e)
            if "TESSDATA_PREFIX" in error_msg or "eng.traineddata" in error_msg:
                logger.error(
                    "Tesseract language data not found. "
                    "Please set TESSDATA_PREFIX environment variable to your tessdata directory. "
                    "Example: set TESSDATA_PREFIX=C:\\Program Files\\Tesseract-OCR\\tessdata"
                )
            raise

        # Debug: Log if we find key text
        text_lower = text.lower()
        if "pandaexpress" in text_lower or "panda" in text_lower:
            logger.info("Found 'panda' in OCR text")
        if "feedback" in text_lower:
            logger.info("Found 'feedback' in OCR text")
        if "survey" in text_lower:
            logger.info("Found 'survey' in OCR text")

        # Find URLs in text
        url = self._find_survey_url(text)
        logger.info(f"URL found: {url}")

        # Also try to extract survey code
        survey_code = self._find_survey_code(text)

        if url:
            return ReceiptData(
                url=url,
                survey_url=url,
                extraction_method="ocr",
                confidence=self.OCR_CONFIDENCE,
                raw_text=text[:500],  # Store first 500 chars
                store_name=self._detect_store_name(text),
            )

        # If we found a survey code but no URL, we might still be able to help
        if survey_code:
            logger.info(f"Found survey code but no URL: {survey_code}")

        return None

    def _find_survey_code(self, text: str) -> Optional[str]:
        """
        Find survey/validation code in text.

        Args:
            text: Text to search.

        Returns:
            Survey code if found, None otherwise.
        """
        if not text:
            return None

        # Common survey code patterns
        # Panda Express: 2125-7062-1388-0079-0312-9406 (6 groups of 4 digits)
        # Walmart: 12 digit numeric
        # Generic: XXXX-XXXXXX format

        patterns = [
            # 6 groups of 4 digits with dashes (Panda Express style)
            r'\b(\d{4}-\d{4}-\d{4}-\d{4}-\d{4}-\d{4})\b',
            # 4 groups of 4 digits with dashes
            r'\b(\d{4}-\d{4}-\d{4}-\d{4})\b',
            # Survey code label followed by digits
            r'(?:survey|validation|code)[:\s]+(\d[\d\-]{10,30})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _detect_store_name(self, text: str) -> Optional[str]:
        """
        Detect store name from receipt text.

        Args:
            text: OCR text from receipt.

        Returns:
            Store name if detected.
        """
        if not text:
            return None

        text_lower = text.lower()

        store_keywords = {
            "Panda Express": ["panda express", "pandaexpress"],
            "Walmart": ["walmart", "wal-mart"],
            "McDonald's": ["mcdonald", "mcdonalds"],
            "Wendy's": ["wendy's", "wendys"],
            "Taco Bell": ["taco bell", "tacobell"],
            "Burger King": ["burger king", "burgerking"],
            "Subway": ["subway"],
            "Starbucks": ["starbucks"],
            "Chick-fil-A": ["chick-fil-a", "chickfila"],
            "Target": ["target"],
        }

        for store_name, keywords in store_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return store_name

        return None

    async def _extract_vision(self, image_path: Path) -> Optional[ReceiptData]:
        """
        Extract URL using Vision API.

        Args:
            image_path: Path to the image file.

        Returns:
            ReceiptData if URL found, None otherwise.
        """
        # Placeholder for Vision API integration
        # Could use Claude Vision, OpenAI Vision, Google Cloud Vision, etc.
        logger.debug("Vision API extraction not implemented")
        return None

    def _is_survey_url(self, text: str) -> bool:
        """
        Check if text looks like a survey URL.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be a survey URL.
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check for known survey domains (even without protocol)
        for domain in KNOWN_SURVEY_DOMAINS:
            if domain.lower() in text_lower:
                return True

        # Check for URL structure
        if text.startswith(("http://", "https://")) or "." in text:
            # Check for survey-related keywords
            survey_keywords = [
                "survey", "feedback", "opinion", "tellus", "voice", "myview",
                "experience", "guest", "visit", "tell", "listen", "talk",
                "satisfaction", "rate", "review",
            ]
            if any(kw in text_lower for kw in survey_keywords):
                return True

            # Check if URL has ?cn= parameter (common survey code format)
            if "?cn=" in text or "&cn=" in text:
                return True

        return False

    def _find_survey_url(self, text: str) -> Optional[str]:
        """
        Find survey URL in text.

        Args:
            text: Text to search.

        Returns:
            Survey URL if found, None otherwise.
        """
        if not text:
            return None

        # First, check for known survey domains (without protocol)
        text_lower = text.lower()
        for domain in KNOWN_SURVEY_DOMAINS:
            if domain.lower() in text_lower:
                # Found a known survey domain, construct full URL
                return f"https://www.{domain}"

        # Try each regex pattern
        for pattern in SURVEY_URL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the URL
                url = match.strip().rstrip(".,;:!?")

                # If it doesn't have protocol, add it
                if not url.startswith(("http://", "https://")):
                    url = f"https://www.{url}"

                # Validate it's a reasonable URL
                if len(url) > 10 and "." in url:
                    return url

        return None

    def find_survey_url(self, text: str) -> Optional[str]:
        """
        Public method to find survey URL in text.

        Useful for testing OCR text parsing separately.

        Args:
            text: Text to search.

        Returns:
            Survey URL if found, None otherwise.
        """
        return self._find_survey_url(text)