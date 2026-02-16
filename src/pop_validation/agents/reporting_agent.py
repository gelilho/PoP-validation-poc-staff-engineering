"""ReportingAgent — Step 7: Persist the result to the audit trail.

Wraps reporting.csv_logger.ResultLogger. Today CSV, tomorrow BigQuery.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pop_validation.agents.base import AgentResult
from pop_validation.models import ImageValidationResult
from pop_validation.reporting.csv_logger import CsvResultLogger, ResultLogger


class ReportingAgent:
    """Log one validation result row to the audit trail."""

    def __init__(self, result_logger: ResultLogger | None = None) -> None:
        self._logger: ResultLogger = result_logger or CsvResultLogger()

    @property
    def name(self) -> str:
        return "Reporting"

    def execute(self, **kwargs: Any) -> AgentResult:
        """Log the result.

        Args:
            result: The ImageValidationResult to persist.
            warranty_id: Warranty ID for correlation.
            image_url: Original image path/URL.
            comment: Optional error description.

        Returns:
            AgentResult (always success — logging never raises).
        """
        result: ImageValidationResult = kwargs["result"]
        warranty_id: str = kwargs.get("warranty_id", "")
        image_url: str = kwargs.get("image_url", "")
        comment: str | None = kwargs.get("comment")

        logger.info("[{}] Logging result for {}", self.name, image_url)

        self._logger.log(
            result=result,
            warranty_id=warranty_id,
            image_url=image_url,
            comment=comment,
        )

        return AgentResult(success=True)
