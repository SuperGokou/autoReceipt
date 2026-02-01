from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


__all__ = [
    "ReceiptData",
    "ExtractionResult",
]


class ReceiptData(BaseModel):
    """
    Extracted receipt information.
    
    Contains the survey URL and metadata extracted from a receipt
    image via QR code, OCR, or vision API.
    
    Attributes:
        url: The survey URL extracted from the receipt.
        store_name: Name of the store (if detected).
        survey_url: Alias for url field.
        survey_code: Survey validation code (if detected).
        extraction_method: How the data was extracted (qr/ocr/vision).
        confidence: Confidence score of extraction (0-1).
        raw_text: Raw OCR text (if applicable).
        extracted_at: When the extraction occurred.
    """
    url: Optional[str] = Field(default=None, description="Survey URL")
    store_name: Optional[str] = Field(default=None, description="Store name")
    survey_url: Optional[str] = Field(default=None, description="Survey URL (alias)")
    survey_code: Optional[str] = Field(default=None, description="Survey validation code")
    extraction_method: str = Field(default="unknown", description="Extraction method")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    raw_text: Optional[str] = Field(default=None, description="Raw OCR text")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction time")
    
    def __str__(self) -> str:
        """String representation."""
        return f"ReceiptData(store={self.store_name}, url={self.url or self.survey_url})"


class ExtractionResult(BaseModel):
    """
    Result of a receipt extraction attempt.
    
    Contains success status, extracted data, and timing information.
    
    Attributes:
        success: Whether extraction succeeded.
        data: Extracted ReceiptData (if successful).
        error: Error message (if failed).
        processing_time_ms: Time taken in milliseconds.
    """
    success: bool = Field(description="Whether extraction succeeded")
    data: Optional[ReceiptData] = Field(default=None, description="Extracted data")
    error: Optional[str] = Field(default=None, description="Error message")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time")
    
    def __str__(self) -> str:
        """String representation."""
        if self.success:
            return f"ExtractionResult(success=True, data={self.data})"
        return f"ExtractionResult(success=False, error={self.error})"
