"""PoP Validation PoC — Proof of Purchase validation for [Insert your brand] warranty claims."""

from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.extraction.analyzer import PopAnalyzer
from pop_validation.pipeline import PopValidationPipeline

__all__ = [
    "JsonFileProductCatalogProvider",
    "PopAnalyzer",
    "PopValidationPipeline",
    "ProductCatalogProvider",
]
__version__ = "1.0.0"
