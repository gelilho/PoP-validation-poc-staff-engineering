"""PoP Validation PoC — Proof of Purchase validation for [Insert your brand] warranty claims."""

from pop_validation.pipeline import PopValidationPipeline
from pop_validation.product_catalog import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)

__all__ = [
    "JsonFileProductCatalogProvider",
    "PopValidationPipeline",
    "ProductCatalogProvider",
]
__version__ = "1.0.0"
