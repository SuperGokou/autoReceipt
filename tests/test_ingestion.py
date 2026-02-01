"""
Test suite for the Ingestion Agent (Module 1).

This module tests:
- QR code extraction
- OCR text extraction
- Vision API fallback (skipped without API key)
- Error handling for invalid inputs
- Pydantic model validation

Run with: pytest tests/test_ingestion.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.survey_bot.agents.ingestion import IngestionAgent
from src.survey_bot.models.receipt import ExtractionResult, ReceiptData


class TestIngestionAgent:
    """Test suite for the IngestionAgent class."""

    @pytest.mark.asyncio
    async def test_qr_extraction_valid_image(self, qr_code_image: Path):
        """Test QR code extraction from valid image."""
        agent = IngestionAgent()
        result = await agent.extract_from_image(qr_code_image)

        assert result.success is True
        assert result.data is not None
        assert result.data.extraction_method == "qr"
        assert "survey.example.com" in str(result.data.url)
        assert result.data.confidence == IngestionAgent.QR_CONFIDENCE
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_ocr_extraction_text_url(self, text_url_image: Path):
        """Test OCR extraction when no QR code present."""
        agent = IngestionAgent()
        result = await agent.extract_from_image(text_url_image)

        assert result.success is True
        assert result.data is not None
        assert result.data.extraction_method == "ocr"
        assert "survey" in str(result.data.url).lower()
        assert result.data.confidence == IngestionAgent.OCR_CONFIDENCE

    @pytest.mark.asyncio
    async def test_extraction_failure_blank_image(self, blank_image: Path):
        """Test graceful failure on blank image."""
        agent = IngestionAgent()
        result = await agent.extract_from_image(blank_image)

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert "No survey URL found" in result.error
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_extraction_failure_file_not_found(self):
        """Test error handling for non-existent file."""
        agent = IngestionAgent()
        result = await agent.extract_from_image("/nonexistent/path/image.jpg")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "File not found" in result.error

    @pytest.mark.asyncio
    async def test_extraction_failure_corrupted_image(self, corrupted_file: Path):
        """Test error handling for corrupted image."""
        agent = IngestionAgent()
        result = await agent.extract_from_image(corrupted_file)

        assert result.success is False
        assert result.error is not None

    def test_confidence_scores_ordering(self):
        """Verify confidence score hierarchy: QR > Vision > OCR."""
        assert IngestionAgent.QR_CONFIDENCE > IngestionAgent.VISION_CONFIDENCE
        assert IngestionAgent.VISION_CONFIDENCE > IngestionAgent.OCR_CONFIDENCE

        # Verify they're all valid probabilities
        assert 0 <= IngestionAgent.OCR_CONFIDENCE <= 1
        assert 0 <= IngestionAgent.VISION_CONFIDENCE <= 1
        assert 0 <= IngestionAgent.QR_CONFIDENCE <= 1

    def test_agent_initialization_default(self):
        """Test default agent initialization."""
        agent = IngestionAgent()
        assert agent.use_vision_api is False
        assert agent.client is None

    def test_agent_initialization_vision_api_requires_key(self):
        """Test that Vision API requires API key."""
        with pytest.raises(ValueError, match="api_key is required"):
            IngestionAgent(use_vision_api=True)

    def test_agent_initialization_with_vision_api(self):
        """Test agent initialization with Vision API enabled."""
        agent = IngestionAgent(use_vision_api=True, api_key="sk-test-key")
        assert agent.use_vision_api is True
        assert agent.client is not None

    @pytest.mark.skip(reason="Requires valid OpenAI API key")
    @pytest.mark.asyncio
    async def test_vision_api_extraction(self, text_url_image: Path):
        """Test Vision API extraction (requires real API key)."""
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        agent = IngestionAgent(use_vision_api=True, api_key=api_key)
        result = await agent.extract_from_image(text_url_image)

        # Vision API might succeed or fail depending on image
        # Just verify it doesn't crash
        assert isinstance(result, ExtractionResult)


class TestExtractionResult:
    """Test suite for ExtractionResult model validation."""

    def test_successful_result_requires_data(self):
        """Test that success=True requires data field."""
        with pytest.raises(ValueError, match="data must be provided"):
            ExtractionResult(success=True, processing_time_ms=100)

    def test_failed_result_requires_error(self):
        """Test that success=False requires error field."""
        with pytest.raises(ValueError, match="error must be provided"):
            ExtractionResult(success=False, processing_time_ms=100)

    def test_valid_successful_result(self):
        """Test creating valid successful result."""
        data = ReceiptData(
            url="https://survey.example.com",
            extraction_method="qr",
            confidence=0.95
        )
        result = ExtractionResult(
            success=True,
            data=data,
            processing_time_ms=150
        )
        assert result.success is True
        assert "survey.example.com" in str(result.data.url)

    def test_valid_failed_result(self):
        """Test creating valid failed result."""
        result = ExtractionResult(
            success=False,
            error="No URL found",
            processing_time_ms=200
        )
        assert result.success is False
        assert result.error == "No URL found"


class TestReceiptData:
    """Test suite for ReceiptData model validation."""

    def test_valid_receipt_data(self):
        """Test creating valid receipt data."""
        data = ReceiptData(
            url="https://survey.walmart.com/abc123",
            store_name="Walmart",
            extraction_method="qr",
            confidence=0.95
        )
        assert data.store_name == "Walmart"
        assert data.extraction_method == "qr"

    def test_confidence_bounds_too_high(self):
        """Test that confidence > 1 is rejected."""
        with pytest.raises(ValueError):
            ReceiptData(
                url="https://example.com",
                extraction_method="qr",
                confidence=1.5
            )

    def test_confidence_bounds_too_low(self):
        """Test that confidence < 0 is rejected."""
        with pytest.raises(ValueError):
            ReceiptData(
                url="https://example.com",
                extraction_method="qr",
                confidence=-0.1
            )

    def test_invalid_url_rejected(self):
        """Test that invalid URLs are rejected."""
        with pytest.raises(ValueError):
            ReceiptData(
                url="not-a-valid-url",
                extraction_method="qr",
                confidence=0.9
            )

    def test_extraction_method_literal(self):
        """Test that only valid extraction methods are accepted."""
        with pytest.raises(ValueError):
            ReceiptData(
                url="https://example.com",
                extraction_method="invalid_method",
                confidence=0.9
            )

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        data = ReceiptData(
            url="https://example.com",
            extraction_method="ocr",
            confidence=0.7
        )
        assert data.store_name is None
        assert data.receipt_id is None
        assert data.raw_text is None