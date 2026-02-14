"""LLM prompt for structured receipt field extraction.

Single responsibility: extract all receipt/invoice fields into structured JSON.
"""

from __future__ import annotations


def build_extraction_prompt(products: list[str]) -> str:
    """
    Build the Gemini extraction prompt with the On Running product catalog.

    Args:
        products: List of official On Running product names for matching.

    Returns:
        Complete prompt string for receipt field extraction.
    """
    products_formatted = "\n".join(f"  - {p}" for p in products)

    return f"""You are an expert OCR system specialized in extracting structured data from receipts and invoices.

Extract ALL available fields from this receipt/invoice image and return them as a JSON object.

## Fields to Extract

- retailer_name: The retailer or store name. Match to a known retailer if possible. Use "other" if the retailer is not a recognized partner.
- retailer_raw: The exact text of the retailer name as it appears on the receipt.
- retailer_location: Physical store location or address. Use "online" for e-commerce purchases.
- purchase_date: Date of purchase in YYYY-MM-DD format.
- product_name: Name(s) of the product(s) purchased.
- transaction_number: Order, transaction, or receipt number.
- price: Total price as a number (no currency symbol).
- currency: Three-letter currency code (e.g., USD, EUR, GBP, CHF, JPY).
- url: Retailer website URL if visible on the receipt.
- address: Full address if visible.
- country: Country where the purchase was made (full name).
- language_category: Two-letter language code of the receipt text (e.g., "en", "de", "ja", "fr").
- receipt_type: One of: "official_receipt_paper", "official_receipt_digital", "invoice", "e_receipt", "order_confirmation", "bank_statement", "bank_transfer_screenshot", "payment_confirmation", "shipping_label", "other"
- image_in_receipt: true if there is a product image on the receipt, false otherwise.
- confidence: Your confidence in the overall extraction accuracy (0.0 to 1.0).

## Product Matching

Match products to the official product catalog below. For each product found:
- product_counts: map product name to quantity purchased
- product_prices: map product name to price paid
- product_skus: map product name to SKU if visible on the receipt

If a product is NOT in the official catalog, use "other" as the key.

Official Product Catalog:
{products_formatted}

## Response Format

Return ONLY a valid JSON object. Use null for any field you cannot determine.

Example:
{{
    "retailer_name": "Foot Locker",
    "retailer_raw": "FOOT LOCKER #12345",
    "retailer_location": "125 Main Street, New York",
    "purchase_date": "2025-01-15",
    "product_name": "Cloud 5",
    "transaction_number": "TXN-98765",
    "price": 149.99,
    "currency": "USD",
    "url": null,
    "address": "125 Main Street, New York, NY 10001",
    "country": "United States",
    "language_category": "en",
    "receipt_type": "official_receipt_paper",
    "image_in_receipt": false,
    "confidence": 0.92,
    "product_counts": {{"Cloud 5": 1}},
    "product_prices": {{"Cloud 5": 149.99}},
    "product_skus": {{}}
}}

Return ONLY the JSON object, no markdown formatting or extra text."""
