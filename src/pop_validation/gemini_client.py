"""Thin Gemini API client — owns model configuration and raw LLM calls.

Single Responsibility: talk to Gemini, return parsed JSON dicts.
No business logic, no domain models, no orchestration.
"""

from __future__ import annotations

import json
import time
from typing import Any

import google.generativeai as genai
from loguru import logger

from pop_validation.config import Settings


class GeminiClient:
    """Low-level Gemini API wrapper.

    Handles model initialization, request serialization, and response parsing.
    Returns raw dicts — callers are responsible for domain interpretation.
    """

    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing GeminiClient...")
        genai.configure(api_key=settings.gemini_api_key)  # type: ignore[attr-defined]
        self._model = genai.GenerativeModel(  # type: ignore[attr-defined]
            model_name=settings.gemini_model_name,
            generation_config=genai.GenerationConfig(  # type: ignore[attr-defined]
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        self._model_name = settings.gemini_model_name
        logger.info("GeminiClient ready | model={}", self._model_name)

    def call(
        self,
        prompt: str,
        image_data: bytes,
        mime_type: str,
        *,
        call_label: str = "GEMINI",
        image_index: int = 0,
    ) -> dict[str, Any]:
        """Send a prompt + image to Gemini and return parsed JSON.

        Args:
            prompt: The text prompt to send.
            image_data: Raw image bytes.
            mime_type: MIME type of the image (e.g. "image/jpeg").
            call_label: Human-readable label for logging (e.g. "Quality Assessment").
            image_index: Image index for log correlation.

        Returns:
            Parsed JSON dict from Gemini response.

        Raises:
            Exception: Any Gemini API or JSON parsing error (fail-fast, no retries).
        """
        logger.info(
            "[{}] {} | sending {} bytes to Gemini...",
            image_index,
            call_label,
            len(image_data),
        )
        start = time.perf_counter()

        image_part = {"mime_type": mime_type, "data": image_data}
        response = self._model.generate_content([prompt, image_part])
        elapsed_ms = (time.perf_counter() - start) * 1000
        data: dict[str, Any] = json.loads(response.text)

        logger.info(
            "[{}] {} DONE | {:.0f}ms",
            image_index,
            call_label,
            elapsed_ms,
        )
        logger.debug("[{}] {} response: {}", image_index, call_label, data)

        return data
