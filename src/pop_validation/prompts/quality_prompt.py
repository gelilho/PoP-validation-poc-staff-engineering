"""LLM prompt for image quality assessment and receipt detection.

Single responsibility: determine if the image is a valid receipt,
assess its quality, and check for AI generation.
"""

from __future__ import annotations

QUALITY_ASSESSMENT_PROMPT = """You are an image quality assessment expert for a warranty claims system.

Analyze the provided image and determine:
1. Is this image a receipt, invoice, or order confirmation?
2. Is the text in the image readable and clear enough to extract data from?
3. Does this image appear to be AI-generated or digitally manipulated?
4. What is the overall image quality?
5. What category does this image belong to?

Return ONLY a valid JSON object with this exact structure:
{
    "is_receipt": true,
    "is_readable": true,
    "is_ai_generated": false,
    "image_quality": "high",
    "image_category": "PROOF_OF_PURCHASE",
    "rejection_reason": null
}

Field definitions:
- is_receipt: true if the image contains a receipt, invoice, order confirmation, or e-receipt
- is_readable: true if the text is clear enough to extract meaningful data
- is_ai_generated: true if the image shows signs of AI generation (artifacts, inconsistencies, synthetic patterns)
- image_quality: one of "high", "medium", or "low"
- image_category: one of "PROOF_OF_PURCHASE", "SHOE_SOLES", "SHOE_INNERTAG", "SHOE_LABEL", "SHOE_DEFECT", "OTHER"
- rejection_reason: null if the image is acceptable, or one of:
  - "LOW_IMAGE_QUALITY" if the image is too blurry, dark, or unreadable
  - "INCORRECT_IMAGE_TYPE" if the image is not a receipt or invoice

Return ONLY the JSON object, no markdown formatting or extra text."""


def build_quality_prompt() -> str:
    """Return the quality assessment prompt."""
    return QUALITY_ASSESSMENT_PROMPT
