"""Shared test fixtures for the PoP validation test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from pop_validation.config import Settings
from pop_validation.models import (
    ImageAnalysis,
    ImageCategory,
    ImageQuality,
    ProductCategory,
    ReceiptFields,
)

TESTS_DIR = Path(__file__).parent
IMAGES_DIR = TESTS_DIR / "images"


@pytest.fixture
def settings() -> Settings:
    """Test settings with a dummy API key."""
    return Settings(
        gemini_api_key="test-api-key-not-real",
        gemini_model_name="gemini-2.5-flash",
    )


@pytest.fixture
def shoe_receipt_path() -> str:
    """Path to a valid shoe receipt test image."""
    return str(IMAGES_DIR / "shoe_receipt.webp")


@pytest.fixture
def blurred_receipt_path() -> str:
    """Path to a blurred receipt test image."""
    return str(IMAGES_DIR / "blurred_receipt.PNG")


@pytest.fixture
def valid_receipt_fields() -> ReceiptFields:
    """A fully valid receipt extraction with all required fields."""
    return ReceiptFields(
        retailer_name="Foot Locker",
        retailer_raw="FOOT LOCKER #12345",
        retailer_location="125 Main Street, New York",
        purchase_date="2025-01-15",
        product_name="Cloud 5",
        transaction_number="TXN-98765",
        price=149.99,
        currency="USD",
        receipt_type="official_receipt_paper",
        product_counts={"Cloud 5": 1},
        product_prices={"Cloud 5": 149.99},
        confidence=0.95,
    )


@pytest.fixture
def valid_image_analysis(valid_receipt_fields: ReceiptFields) -> ImageAnalysis:
    """An ImageAnalysis that passes all validation rules."""
    return ImageAnalysis(
        image_index=0,
        image_url="https://example.com/receipt.jpg",
        image_quality=ImageQuality.HIGH,
        image_category=ImageCategory.PROOF_OF_PURCHASE,
        product_category=ProductCategory.SHOES,
        language_category="en",
        is_ai_generated=False,
        receipt_fields=valid_receipt_fields,
    )
