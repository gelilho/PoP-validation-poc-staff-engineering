"""Pydantic models for all data structures in the PoP validation pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class ReceiptType(StrEnum):
    """All possible receipt/document types."""

    OFFICIAL_RECEIPT_PAPER = "official_receipt_paper"
    OFFICIAL_RECEIPT_DIGITAL = "official_receipt_digital"
    INVOICE = "invoice"
    E_RECEIPT = "e_receipt"
    ORDER_CONFIRMATION = "order_confirmation"
    BANK_STATEMENT = "bank_statement"
    BANK_TRANSFER_SCREENSHOT = "bank_transfer_screenshot"
    PAYMENT_CONFIRMATION = "payment_confirmation"
    SHIPPING_LABEL = "shipping_label"
    OTHER = "other"


class ImageCategory(StrEnum):
    """Classification of what the image contains."""

    PROOF_OF_PURCHASE = "PROOF_OF_PURCHASE"
    SHOE_SOLES = "SHOE_SOLES"
    SHOE_INNERTAG = "SHOE_INNERTAG"
    SHOE_LABEL = "SHOE_LABEL"
    SHOE_DEFECT = "SHOE_DEFECT"
    OTHER = "OTHER"


class ProductCategory(StrEnum):
    """[Insert your brand] product categories."""

    SHOES = "shoes"
    ACCESSORIES = "accessories"
    APPAREL = "apparel"


class ImageQuality(StrEnum):
    """Image quality assessment levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ──────────────────────────────────────────────
# Input Models
# ──────────────────────────────────────────────


class ValidationRequest(BaseModel):
    """Input to the PoP validation pipeline."""

    warranty_id: str
    image_urls: list[str] = Field(..., min_length=1, max_length=10)


# ──────────────────────────────────────────────
# Internal Models
# ──────────────────────────────────────────────


class ImageQualityReport(BaseModel):
    """Result of technical + semantic image quality assessment."""

    resolution_ok: bool = True
    blur_score: float = 0.0
    is_blurry: bool = False
    format_ok: bool = True
    file_size_ok: bool = True
    is_receipt: bool = False
    is_readable: bool = True
    is_ai_generated: bool = False
    image_quality: ImageQuality = ImageQuality.HIGH
    image_category: ImageCategory = ImageCategory.OTHER
    rejection_reason: str | None = None


class ReceiptFields(BaseModel):
    """Fields extracted from a receipt/invoice image by Gemini OCR."""

    retailer_name: str | None = None
    retailer_raw: str | None = None
    retailer_location: str | None = None
    purchase_date: str | None = None  # YYYY-MM-DD
    product_name: str | list[str] | None = None
    transaction_number: str | None = None
    price: float | None = None
    currency: str | None = None
    url: str | None = None
    address: str | None = None
    country: str | None = None
    language_category: str | None = None
    product_sku: str | dict[str, str] | None = None
    product_counts: dict[str, int] | None = None
    product_prices: dict[str, float] | None = None
    product_skus: dict[str, str] | None = None
    image_in_receipt: bool | None = None
    confidence: float | None = None
    receipt_type: str | None = None


class ImageAnalysis(BaseModel):
    """Complete analysis result for a single image."""

    image_index: int
    image_url: str
    image_quality: ImageQuality | None = None
    image_category: ImageCategory | None = None
    product_category: ProductCategory | None = None
    language_category: str | None = None
    is_ai_generated: bool = False
    receipt_fields: ReceiptFields | None = None


# ──────────────────────────────────────────────
# Output Models
# ──────────────────────────────────────────────


class ImageValidationResult(BaseModel):
    """Validation result for a single image, with all extracted fields flattened."""

    message: str
    image_category: ImageCategory | None = None
    product_category: ProductCategory | None = None
    language_category: str | None = None
    is_ai_generated: bool = False
    retailer_name: str | None = None
    retailer_raw: str | None = None
    retailer_location: str | None = None
    purchase_date: str | None = None
    product_name: str | list[str] | None = None
    transaction_number: str | None = None
    price: float | None = None
    currency: str | None = None
    url: str | None = None
    address: str | None = None
    country: str | None = None
    product_sku: str | dict[str, str] | None = None
    product_counts: dict[str, int] | None = None
    product_prices: dict[str, float] | None = None
    product_skus: dict[str, str] | None = None
    image_in_receipt: bool | None = None
    confidence: float | None = None
    receipt_type: str | None = None


class ValidationResponse(BaseModel):
    """Final output of the PoP validation pipeline."""

    pop_valid: bool = False
    uncertain: bool = False
    pop_validation_results: list[ImageValidationResult] = Field(default_factory=list)
