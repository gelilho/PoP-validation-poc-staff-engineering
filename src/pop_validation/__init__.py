"""PoP Validation PoC — Proof of Purchase validation for [Insert your brand] warranty claims."""

from pop_validation.agents.orchestrator import OrchestratorAgent
from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.pipeline import PopValidationPipeline
from pop_validation.reporting.csv_logger import CsvResultLogger, ResultLogger

__all__ = [
    "CsvResultLogger",
    "JsonFileProductCatalogProvider",
    "OrchestratorAgent",
    "PopValidationPipeline",
    "ProductCatalogProvider",
    "ResultLogger",
]
__version__ = "2.0.0"
