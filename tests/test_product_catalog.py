"""Tests for product catalog abstraction — Protocol + default JSON implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
)


class TestJsonFileProductCatalogProvider:
    def test_loads_from_default_path(self) -> None:
        provider = JsonFileProductCatalogProvider()
        products = provider.get_products()
        assert isinstance(products, list)
        assert len(products) >= 60

    def test_contains_known_products(self) -> None:
        provider = JsonFileProductCatalogProvider()
        products = provider.get_products()
        assert "Cloud 5" in products
        assert "Cloudsurfer" in products

    def test_loads_from_custom_path(self, tmp_path: Path) -> None:
        catalog_file = tmp_path / "custom.json"
        catalog_file.write_text(json.dumps(["ProductA", "ProductB"]))
        provider = JsonFileProductCatalogProvider(path=catalog_file)
        assert provider.get_products() == ["ProductA", "ProductB"]

    def test_caches_result(self) -> None:
        provider = JsonFileProductCatalogProvider()
        first = provider.get_products()
        second = provider.get_products()
        assert first is second  # Same object reference = cache hit

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        provider = JsonFileProductCatalogProvider(path=tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            provider.get_products()


class TestProductCatalogProtocol:
    def test_custom_provider_satisfies_protocol(self) -> None:
        """Any class with get_products() -> list[str] works (structural typing)."""

        class InMemoryProvider:
            def get_products(self) -> list[str]:
                return ["TestProduct"]

        provider = InMemoryProvider()
        assert provider.get_products() == ["TestProduct"]
