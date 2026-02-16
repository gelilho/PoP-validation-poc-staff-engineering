"""QualityAgent — Steps 2+3: Technical check (Pillow) + Quality assessment (Gemini #1).

Both answer the same question: "Is this image good enough to extract data from?"
Wraps imaging.validator + client.gemini_client. Zero duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pop_validation.agents.base import AgentResult
from pop_validation.client.gemini_client import GeminiClient
from pop_validation.config import Settings, get_settings
from pop_validation.extraction.field_utils import build_quality_report
from pop_validation.imaging.loader import LoadedImage
from pop_validation.imaging.validator import validate_image as check_technical_quality
from pop_validation.models import ImageQualityReport
from pop_validation.prompts.quality_prompt import build_quality_prompt


@dataclass(frozen=True)
class QualityVerdict:
    """Output of the QualityAgent — did the image pass both checks?

    Attributes:
        passed: True if image is good enough for field extraction.
        technical_ok: True if Pillow checks passed (resolution, blur, format).
        report: The full quality report (technical + semantic fields).
    """

    passed: bool
    technical_ok: bool
    report: ImageQualityReport


class QualityAgent:
    """Assess image quality — fast Pillow checks, then Gemini semantic check."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = GeminiClient(self._settings)
        self._quality_prompt = build_quality_prompt()

    @property
    def name(self) -> str:
        return "Quality"

    def execute(self, **kwargs: Any) -> AgentResult:
        """Run technical + semantic quality checks.

        Args:
            loaded: A LoadedImage from ImageLoaderAgent.
            image_index: Index of the image in the batch.

        Returns:
            AgentResult with data=QualityVerdict.
        """
        loaded: LoadedImage = kwargs["loaded"]
        image_index: int = kwargs.get("image_index", 0)

        logger.info("[{}] Starting quality checks for image [{}]", self.name, image_index)

        # Step 2: Technical validation (free — Pillow)
        try:
            tech = check_technical_quality(loaded)
        except Exception as exc:
            error_msg = f"Technical check failed: {type(exc).__name__}: {exc}"
            logger.error("[{}] {}", self.name, error_msg)
            return AgentResult(success=False, error=error_msg)

        if not tech.resolution_ok or not tech.file_size_ok:
            logger.warning(
                "[{}] REJECTED by technical validation | Gemini SKIPPED",
                self.name,
            )
            return AgentResult(
                success=True,
                data=QualityVerdict(passed=False, technical_ok=False, report=tech),
            )

        # Step 3: Quality assessment (Gemini call #1)
        try:
            raw = self._client.call(
                prompt=self._quality_prompt,
                image_data=loaded.data,
                mime_type=loaded.mime_type,
                call_label="Quality Assessment",
                image_index=image_index,
            )
            report = build_quality_report(raw)
        except Exception as exc:
            logger.error("[{}] Gemini quality assessment failed: {}", self.name, exc)
            report = ImageQualityReport(
                rejection_reason=f"QUALITY_ASSESSMENT_FAILED: {type(exc).__name__}",
            )

        passed = report.rejection_reason is None
        if not passed:
            logger.warning(
                "[{}] REJECTED by quality assessment | Extraction SKIPPED",
                self.name,
            )

        return AgentResult(
            success=True,
            data=QualityVerdict(passed=passed, technical_ok=True, report=report),
        )
