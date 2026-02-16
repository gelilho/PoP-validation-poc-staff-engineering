"""ValidationAgent — Steps 5+6: Apply business rules and build the final result.

Pure logic — no I/O, no API calls. Wraps validation.rules.apply_rules().
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pop_validation.agents.base import AgentResult
from pop_validation.models import ImageAnalysis, ImageValidationResult
from pop_validation.validation.rules import apply_rules


class ValidationAgent:
    """Apply the 9-rule chain of responsibility and build ImageValidationResult."""

    @property
    def name(self) -> str:
        return "Validation"

    def execute(self, **kwargs: Any) -> AgentResult:
        """Apply rules and build the result.

        Args:
            analysis: A fully populated ImageAnalysis.

        Returns:
            AgentResult with data=ImageValidationResult.
        """
        analysis: ImageAnalysis = kwargs["analysis"]

        logger.info("[{}] Applying rules for image [{}]", self.name, analysis.image_index)

        try:
            message, _pop_valid, _uncertain = apply_rules(analysis)
            logger.info("[{}] Rules -> {}", self.name, message)

            result = _build_result(analysis, message)
            return AgentResult(success=True, data=result)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("[{}] Rule application failed: {}", self.name, error_msg)
            return AgentResult(success=False, error=error_msg)


def _build_result(analysis: ImageAnalysis, message: str) -> ImageValidationResult:
    """Build an ImageValidationResult by merging metadata with receipt fields."""
    result_data: dict[str, object] = {}

    if analysis.receipt_fields is not None:
        result_data.update(analysis.receipt_fields.model_dump())

    result_data["message"] = message
    result_data["image_category"] = analysis.image_category
    result_data["product_category"] = analysis.product_category
    result_data["is_ai_generated"] = analysis.is_ai_generated
    if analysis.language_category is not None:
        result_data["language_category"] = analysis.language_category

    return ImageValidationResult(**result_data)  # type: ignore[arg-type, unused-ignore]
