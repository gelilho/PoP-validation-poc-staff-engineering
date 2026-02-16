"""OrchestratorAgent — coordinates all agents for the full validation pipeline.

This is the brain. It receives a ValidationRequest, runs each image through
the 5 specialist agents in order, handles early exits, and aggregates results.

Agent flow per image:
    ImageLoaderAgent → QualityAgent → ExtractionAgent → ValidationAgent → ReportingAgent
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from pop_validation.agents.base import AgentResult
from pop_validation.agents.extraction_agent import ExtractionAgent
from pop_validation.agents.image_loader_agent import ImageLoaderAgent
from pop_validation.agents.quality_agent import QualityAgent, QualityVerdict
from pop_validation.agents.reporting_agent import ReportingAgent
from pop_validation.agents.validation_agent import ValidationAgent
from pop_validation.catalog.provider import (
    JsonFileProductCatalogProvider,
    ProductCatalogProvider,
)
from pop_validation.config import Settings, get_settings
from pop_validation.extraction.field_utils import (
    get_language,
    infer_product_category,
)
from pop_validation.models import (
    ImageAnalysis,
    ImageQuality,
    ImageValidationResult,
    ReceiptFields,
    ValidationRequest,
    ValidationResponse,
)
from pop_validation.reporting.csv_logger import ResultLogger


class OrchestratorAgent:
    """Coordinates all 5 specialist agents.

    Usage:
        orchestrator = OrchestratorAgent()
        response = orchestrator.run(ValidationRequest(
            warranty_id="W-123",
            image_urls=["receipt.jpg"],
        ))
    """

    def __init__(
        self,
        settings: Settings | None = None,
        product_catalog: ProductCatalogProvider | None = None,
        result_logger: ResultLogger | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        catalog = product_catalog or JsonFileProductCatalogProvider()

        # Create all 5 specialist agents
        self._loader = ImageLoaderAgent()
        self._quality = QualityAgent(settings=resolved_settings)
        self._extraction = ExtractionAgent(
            settings=resolved_settings,
            product_catalog=catalog,
        )
        self._validation = ValidationAgent()
        self._reporting = ReportingAgent(result_logger=result_logger)

        logger.info(
            "OrchestratorAgent ready | 5 agents initialized | catalog={} products",
            len(catalog.get_products()),
        )

    @property
    def name(self) -> str:
        return "Orchestrator"

    # ── Public API ────────────────────────────────

    def run(self, request: ValidationRequest) -> ValidationResponse:
        """Validate all images and aggregate results.

        For each image → run_image() (5 agents in sequence).
        Then OR-aggregate: any valid → pop_valid=True.
        """
        start = time.perf_counter()
        logger.info(
            "ORCHESTRATOR START | warranty_id={} | images={}",
            request.warranty_id,
            len(request.image_urls),
        )

        pop_valid = False
        uncertain = False
        results: list[ImageValidationResult] = []

        for idx, url in enumerate(request.image_urls):
            result = self.run_image(url, idx, request.warranty_id)
            results.append(result)

            if result.message == "VALID_RECEIPT_FOUND":
                pop_valid = True
            if result.message and "MISSING_REQUIRED" in result.message:
                uncertain = True

        response = ValidationResponse(
            pop_valid=pop_valid,
            uncertain=uncertain,
            pop_validation_results=results,
        )

        self._log_summary(request, response, start)
        return response

    def run_image(
        self,
        image_url: str,
        image_index: int,
        warranty_id: str = "",
    ) -> ImageValidationResult:
        """Run the 5 agents in sequence for a single image.

        Agent 1: ImageLoaderAgent  → load image
        Agent 2: QualityAgent      → tech + Gemini quality check
        Agent 3: ExtractionAgent   → Gemini field extraction (skipped if quality fails)
        Agent 4: ValidationAgent   → apply rules + build result
        Agent 5: ReportingAgent    → persist to audit trail
        """
        logger.info("--- Image [{}] | source={}", image_index, image_url)
        comment: str | None = None

        # Agent 1: Load image
        load_result = self._loader.execute(image_url=image_url)
        if not load_result.success:
            comment = load_result.error
            return self._handle_failure(
                image_index, image_url, warranty_id, comment,
            )

        loaded = load_result.data

        # Agent 2: Quality check (tech + Gemini #1)
        quality_result = self._quality.execute(
            loaded=loaded, image_index=image_index,
        )
        if not quality_result.success:
            comment = quality_result.error
            return self._handle_failure(
                image_index, image_url, warranty_id, comment,
            )

        verdict: QualityVerdict = quality_result.data

        if not verdict.passed:
            # Quality failed — skip extraction, go straight to rules
            analysis = self._build_rejection_analysis(
                image_index, image_url, verdict,
            )
            return self._validate_and_report(
                analysis, warranty_id, image_url, comment,
            )

        # Agent 3: Field extraction (Gemini #2 — only if quality passed)
        extraction_result = self._extraction.execute(
            loaded=loaded, image_index=image_index,
        )
        fields: ReceiptFields | None = extraction_result.data
        if extraction_result.error:
            comment = extraction_result.error

        # Build full analysis
        analysis = self._build_full_analysis(
            image_index, image_url, verdict, fields,
        )

        # Agent 4 + 5: Validate and report
        return self._validate_and_report(
            analysis, warranty_id, image_url, comment,
        )

    # ── Internal helpers ──────────────────────────

    def _validate_and_report(
        self,
        analysis: ImageAnalysis,
        warranty_id: str,
        image_url: str,
        comment: str | None,
    ) -> ImageValidationResult:
        """Agent 4 (ValidationAgent) + Agent 5 (ReportingAgent)."""
        # Agent 4: Apply rules + build result
        validation_result = self._validation.execute(analysis=analysis)
        result: ImageValidationResult = validation_result.data

        # Agent 5: Log to audit trail
        self._reporting.execute(
            result=result,
            warranty_id=warranty_id,
            image_url=image_url,
            comment=comment,
        )

        return result

    def _handle_failure(
        self,
        image_index: int,
        image_url: str,
        warranty_id: str,
        comment: str | None,
    ) -> ImageValidationResult:
        """Handle agent failure — create empty analysis, run rules, report."""
        analysis = ImageAnalysis(image_index=image_index, image_url=image_url)
        return self._validate_and_report(
            analysis, warranty_id, image_url, comment,
        )

    @staticmethod
    def _build_rejection_analysis(
        idx: int,
        url: str,
        verdict: QualityVerdict,
    ) -> ImageAnalysis:
        """Build an ImageAnalysis for a quality-rejected image."""
        if not verdict.technical_ok:
            return ImageAnalysis(
                image_index=idx,
                image_url=url,
                image_quality=ImageQuality.LOW,
            )
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=verdict.report.image_quality,
            image_category=verdict.report.image_category,
            is_ai_generated=verdict.report.is_ai_generated,
        )

    @staticmethod
    def _build_full_analysis(
        idx: int,
        url: str,
        verdict: QualityVerdict,
        fields: ReceiptFields | None,
    ) -> ImageAnalysis:
        """Build a complete ImageAnalysis after successful extraction."""
        return ImageAnalysis(
            image_index=idx,
            image_url=url,
            image_quality=verdict.report.image_quality,
            image_category=verdict.report.image_category,
            product_category=infer_product_category(fields),
            language_category=get_language(verdict.report, fields),
            is_ai_generated=verdict.report.is_ai_generated,
            receipt_fields=fields,
        )

    @staticmethod
    def _log_summary(
        request: ValidationRequest,
        response: ValidationResponse,
        start: float,
    ) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "ORCHESTRATOR DONE | warranty_id={} | pop_valid={} | uncertain={} | "
            "images={} | {:.0f}ms",
            request.warranty_id,
            response.pop_valid,
            response.uncertain,
            len(response.pop_validation_results),
            elapsed_ms,
        )
        for i, r in enumerate(response.pop_validation_results):
            logger.info("  [{}] {} | retailer={}", i, r.message, r.retailer_name)

    def execute(self, **kwargs: Any) -> AgentResult:
        """BaseAgent interface — delegates to run()."""
        request: ValidationRequest = kwargs["request"]
        response = self.run(request)
        return AgentResult(success=True, data=response)
