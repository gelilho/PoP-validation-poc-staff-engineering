"""Product catalog abstraction — Strategy pattern for injectable data sources.

Defines a Protocol for product catalog providers and a default JSON file implementation.
Swap in any provider (API, database, CMS) with zero code changes to the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from loguru import logger

_DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "products.json"


class ProductCatalogProvider(Protocol):
    """Interface for product catalog data sources.

    Implement this protocol to provide products from any backend:
    - JSON file (default, included)
    - REST API
    - Database query
    - CMS / PIM system
    """

    def get_products(self) -> list[str]:
        """Return the current product catalog as a list of product names."""
        ...


class JsonFileProductCatalogProvider:
    """Default implementation — reads product names from a local JSON file.

    The file is read once and cached for the lifetime of the provider instance.
    To refresh, create a new instance.

    Args:
        path: Path to the JSON file. Defaults to the bundled products.json.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_CATALOG_PATH
        self._cache: list[str] | None = None

    def get_products(self) -> list[str]:
        """Load products from JSON file (cached after first read)."""
        if self._cache is None:
            logger.debug("Loading product catalog from: {}", self._path)
            with open(self._path) as f:
                self._cache = json.load(f)
            logger.info(
                "Product catalog loaded | {} products | source={}",
                len(self._cache),
                self._path,
            )
        return self._cache
