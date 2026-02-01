from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


__all__ = [
    "CouponPattern",
    "CouponResult",
    "SurveyResult",
    "EmailDeliveryResult",
    "ExtractionMethod",
]


class ExtractionMethod(str, Enum):
    """How the coupon was extracted."""
    REGEX_PATTERN = "regex_pattern"      # Matched a known pattern
    ELEMENT_SEARCH = "element_search"    # Found in specific element
    OCR = "ocr"                          # Optical character recognition
    LLM_EXTRACTION = "llm_extraction"    # LLM identified the code
    MANUAL = "manual"                    # Manually provided


class CouponPattern(str, Enum):
    """
    Known coupon code patterns from various survey providers.
    """
    # Generic patterns
    ALPHA_NUMERIC = "alpha_numeric"      # ABC123456
    ALPHA_DASH_NUMERIC = "alpha_dash_numeric"  # ABCD-123456
    NUMERIC_ONLY = "numeric_only"        # 123456789
    
    # Specific survey providers
    WALMART = "walmart"                  # 12 digit numeric
    MCDONALDS = "mcdonalds"              # Various formats
    TACO_BELL = "taco_bell"              # Survey code format
    WENDYS = "wendys"                    # Validation code
    GENERIC = "generic"                  # Unknown format


class CouponResult(BaseModel):
    """
    Result of coupon extraction from a survey completion page.
    
    Contains the extracted code, proof screenshot, and metadata
    about how and when the code was extracted.
    
    Attributes:
        code: The extracted coupon/validation code.
        screenshot_path: Path to screenshot of the coupon page.
        extracted_at: When the coupon was extracted.
        confidence: Confidence score of extraction (0-1).
        pattern_type: Type of coupon pattern detected.
        extraction_method: How the coupon was found.
        raw_text: Raw text that contained the code (for debugging).
    """
    code: str = Field(description="Extracted coupon/validation code")
    screenshot_path: Optional[str] = Field(
        default=None,
        description="Path to proof screenshot"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="Extraction timestamp"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction accuracy"
    )
    pattern_type: CouponPattern = Field(
        default=CouponPattern.GENERIC,
        description="Detected coupon pattern type"
    )
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.REGEX_PATTERN,
        description="Method used to extract coupon"
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Raw text containing the code"
    )
    
    model_config = {"use_enum_values": True}
    
    @field_validator("code")
    @classmethod
    def normalize_code(cls, v: str) -> str:
        """Normalize the coupon code."""
        # Remove extra whitespace and convert to uppercase
        return v.strip().upper()
    
    def __str__(self) -> str:
        """String representation."""
        return f"Coupon({self.code})"
    
    def to_display_format(self) -> str:
        """Format code for display (with dashes if needed)."""
        code = self.code.replace("-", "").replace(" ", "")
        
        # Format based on length
        if len(code) == 12:
            return f"{code[:4]}-{code[4:8]}-{code[8:]}"
        elif len(code) == 10:
            return f"{code[:4]}-{code[4:]}"
        elif len(code) == 8:
            return f"{code[:4]}-{code[4:]}"
        
        return self.code


class SurveyResult(BaseModel):
    """
    Complete result of a survey automation run.
    
    Contains all information about a survey completion attempt,
    including success status, extracted coupon, and timing data.
    
    Attributes:
        success: Whether survey was completed successfully.
        coupon: Extracted coupon result (if successful).
        email_sent: Whether coupon was sent via email.
        survey_url: Original survey URL.
        receipt_image_path: Path to original receipt image.
        start_time: When automation started.
        end_time: When automation completed.
        steps_taken: Number of navigation steps taken.
        actions_log: Log of all actions taken.
        error_message: Error message (if failed).
    """
    success: bool = Field(description="Whether survey completed successfully")
    coupon: Optional[CouponResult] = Field(
        default=None,
        description="Extracted coupon (if successful)"
    )
    email_sent: bool = Field(
        default=False,
        description="Whether coupon was sent via email"
    )
    survey_url: str = Field(description="Survey URL that was completed")
    receipt_image_path: Optional[str] = Field(
        default=None,
        description="Original receipt image path"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Automation start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Automation end time"
    )
    steps_taken: int = Field(default=0, description="Navigation steps taken")
    actions_log: List[str] = Field(
        default_factory=list,
        description="Log of actions taken"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    model_config = {"use_enum_values": True}
    
    @property
    def coupon_code(self) -> Optional[str]:
        """Get the coupon code (convenience property)."""
        return self.coupon.code if self.coupon else None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration of the survey in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_summary(self) -> dict:
        """Get a summary dictionary."""
        return {
            "success": self.success,
            "coupon_code": self.coupon.code if self.coupon else None,
            "survey_url": self.survey_url,
            "steps_taken": self.steps_taken,
            "duration_seconds": self.duration_seconds,
            "error": self.error_message,
        }


class EmailDeliveryResult(BaseModel):
    """
    Result of sending a coupon via email.
    
    Attributes:
        success: Whether email was sent successfully.
        recipient: Email recipient address.
        sent_at: When email was sent.
        message_id: Email message ID (if available).
        error_message: Error message (if failed).
    """
    success: bool = Field(description="Whether email was sent")
    recipient: str = Field(description="Recipient email address")
    sent_at: datetime = Field(
        default_factory=datetime.now,
        description="When email was sent"
    )
    message_id: Optional[str] = Field(
        default=None,
        description="Email message ID"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    coupon_code: Optional[str] = Field(
        default=None,
        description="Coupon code that was sent"
    )
