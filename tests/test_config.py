"""Tests for application configuration and constants."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from pop_validation.config import (
    REQUIRED_FIELDS,
    VALID_RECEIPT_TYPES,
    Settings,
    load_products,
)


class TestSettings:
    def test_settings_from_env_gemini_api_key(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.gemini_api_key == "test-key-123"
            assert settings.gemini_model_name == "gemini-2.5-flash"

    def test_settings_from_env_google_api_key_compat(self) -> None:
        """GOOGLE_API_KEY is accepted as a backwards-compatible alias."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "legacy-key-456"}):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.gemini_api_key == "legacy-key-456"

    def test_settings_custom_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL_NAME": "gemini-2.0-pro",
            },
        ):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.gemini_model_name == "gemini-2.0-pro"

    def test_settings_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True), pytest.raises((ValueError, KeyError)):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_settings_direct_init(self, settings: Settings) -> None:
        assert settings.gemini_api_key == "test-api-key-not-real"


class TestConstants:
    def test_valid_receipt_types_count(self) -> None:
        assert len(VALID_RECEIPT_TYPES) == 5

    def test_valid_receipt_types_contents(self) -> None:
        assert "official_receipt_paper" in VALID_RECEIPT_TYPES
        assert "invoice" in VALID_RECEIPT_TYPES
        assert "e_receipt" in VALID_RECEIPT_TYPES
        assert "bank_statement" not in VALID_RECEIPT_TYPES

    def test_required_fields_count(self) -> None:
        assert len(REQUIRED_FIELDS) == 5

    def test_required_fields_contents(self) -> None:
        assert "retailer_name" in REQUIRED_FIELDS
        assert "purchase_date" in REQUIRED_FIELDS
        assert "product_counts" in REQUIRED_FIELDS


class TestLoadProducts:
    def test_loads_products(self) -> None:
        products = load_products()
        assert isinstance(products, list)
        assert len(products) > 0

    def test_contains_known_products(self) -> None:
        products = load_products()
        assert "Cloud 5" in products
        assert "Cloudsurfer" in products

    def test_product_count(self) -> None:
        products = load_products()
        assert len(products) >= 60  # Catalog may grow; ensure reasonable count
