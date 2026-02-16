"""ImageLoaderAgent — Step 1: Load an image from any source.

Wraps imaging.loader.load_image(). Zero duplication.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pop_validation.agents.base import AgentResult
from pop_validation.imaging.loader import load_image


class ImageLoaderAgent:
    """Load an image from a URL, file path, or file:// URI."""

    @property
    def name(self) -> str:
        return "ImageLoader"

    def execute(self, **kwargs: Any) -> AgentResult:
        """Load the image.

        Args:
            image_url: Path or URL to the image.

        Returns:
            AgentResult with data=LoadedImage on success.
        """
        image_url: str = kwargs["image_url"]
        logger.info("[{}] Loading image: {}", self.name, image_url)

        try:
            loaded = load_image(image_url)
            logger.info(
                "[{}] Loaded | {} bytes | {}",
                self.name,
                len(loaded.data),
                loaded.mime_type,
            )
            return AgentResult(success=True, data=loaded)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("[{}] Failed: {}", self.name, error_msg)
            return AgentResult(success=False, error=error_msg)
