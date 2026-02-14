"""Load images from URLs, file:// URIs, or local file paths.

Factory Method pattern: dispatches to the right loader based on input scheme.
"""

from __future__ import annotations

import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger
from PIL import Image


@dataclass(frozen=True)
class LoadedImage:
    """An image loaded into memory, ready for validation and Gemini."""

    data: bytes
    mime_type: str
    source: str
    pil_image: Image.Image


def load_image(image_url_or_path: str, timeout: int = 30) -> LoadedImage:
    """
    Load an image from a URL, file:// URI, or local file path.

    Args:
        image_url_or_path: The image source (http://, https://, file://, or local path).
        timeout: HTTP request timeout in seconds.

    Returns:
        LoadedImage with raw bytes, MIME type, and PIL Image.

    Raises:
        ValueError: If the source is empty.
        FileNotFoundError: If a local file does not exist.
        requests.RequestException: If an HTTP download fails.
    """
    if not image_url_or_path or not image_url_or_path.strip():
        logger.error("Image source is empty or blank")
        raise ValueError("Image source cannot be empty")

    source = image_url_or_path.strip()
    logger.debug("Loading image | source={}", source)

    if source.startswith("file://"):
        file_path = source[7:]  # Strip "file://"
        logger.debug("Detected file:// URI, loading from local path: {}", file_path)
        return _load_from_file(file_path, source)

    if source.startswith(("http://", "https://")):
        logger.debug("Detected HTTP URL, downloading with timeout={}s: {}", timeout, source)
        return _load_from_url(source, timeout)

    # Assume local file path
    logger.debug("Detected local file path: {}", source)
    return _load_from_file(source, source)


def _load_from_file(file_path: str, source: str) -> LoadedImage:
    """Load image bytes from a local file."""
    path = Path(file_path)
    if not path.exists():
        logger.error("Image file not found: {}", file_path)
        raise FileNotFoundError(f"Image file not found: {file_path}")

    mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    data = path.read_bytes()
    pil_image = Image.open(io.BytesIO(data))

    logger.info(
        "Image loaded from file | {} bytes | mime={} | dimensions={}x{} | path={}",
        len(data),
        mime_type,
        pil_image.width,
        pil_image.height,
        file_path,
    )
    return LoadedImage(data=data, mime_type=mime_type, source=source, pil_image=pil_image)


def _load_from_url(url: str, timeout: int) -> LoadedImage:
    """Download image bytes from an HTTP/HTTPS URL."""
    logger.info("Downloading image from URL: {}", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    mime_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    data = response.content
    pil_image = Image.open(io.BytesIO(data))

    logger.info(
        "Image downloaded from URL | {} bytes | mime={} | dimensions={}x{} | status={} | url={}",
        len(data),
        mime_type,
        pil_image.width,
        pil_image.height,
        response.status_code,
        url,
    )
    return LoadedImage(data=data, mime_type=mime_type, source=url, pil_image=pil_image)
