"""PoP Validation Pipeline — thin facade over the multi-agent architecture.

The real work happens in the agents/ package. This class preserves backward
compatibility so existing callers (run_pipeline.py, tests) keep working.

Agents (divide and conquer):
    1. ImageLoaderAgent   — load image from any source
    2. QualityAgent       — tech check (Pillow) + quality assessment (Gemini #1)
    3. ExtractionAgent    — field extraction (Gemini #2)
    4. ValidationAgent    — apply 9 business rules + build result
    5. ReportingAgent     — persist to audit trail (CSV / BQ / Postgres)
    +  OrchestratorAgent  — coordinates all agents per image, aggregates results
"""

from __future__ import annotations

from pop_validation.agents.orchestrator import OrchestratorAgent
from pop_validation.catalog.provider import ProductCatalogProvider
from pop_validation.config import Settings
from pop_validation.models import (
    ImageValidationResult,
    ValidationRequest,
    ValidationResponse,
)
from pop_validation.reporting.csv_logger import ResultLogger


class PopValidationPipeline:
    """End-to-end Proof of Purchase validation.

    Facade over the OrchestratorAgent. Same API as before — zero breaking changes.

    Usage:
        pipeline = PopValidationPipeline()
        response = pipeline.validate(ValidationRequest(
            warranty_id="W-123",
            image_urls=["https://example.com/receipt.jpg"],
        ))
    """

    def __init__(
        self,
        settings: Settings | None = None,
        product_catalog: ProductCatalogProvider | None = None,
        result_logger: ResultLogger | None = None,
    ) -> None:
        self._orchestrator = OrchestratorAgent(
            settings=settings,
            product_catalog=product_catalog,
            result_logger=result_logger,
        )

    def validate(self, request: ValidationRequest) -> ValidationResponse:
        """Validate all images and aggregate results.

        Delegates to OrchestratorAgent.run() which coordinates the 5 agents.
        """
        return self._orchestrator.run(request)

    def validate_image(
        self,
        image_url: str,
        image_index: int,
        warranty_id: str = "",
    ) -> ImageValidationResult:
        """Validate a single image through the 5-agent pipeline."""
        return self._orchestrator.run_image(image_url, image_index, warranty_id)
