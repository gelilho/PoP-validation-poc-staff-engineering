"""Tests for image loading from various sources — Factory Method."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pop_validation.imaging.loader import LoadedImage, load_image


class TestLoadImageFromFile:
    def test_load_from_local_path(self, shoe_receipt_path: str) -> None:
        result = load_image(shoe_receipt_path)
        assert isinstance(result, LoadedImage)
        assert len(result.data) > 0
        assert result.mime_type == "image/webp"
        assert result.pil_image is not None

    def test_load_from_file_uri(self, shoe_receipt_path: str) -> None:
        file_uri = f"file://{shoe_receipt_path}"
        result = load_image(file_uri)
        assert isinstance(result, LoadedImage)
        assert len(result.data) > 0

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.jpg")

    def test_file_uri_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_image("file:///nonexistent/path/image.jpg")


class TestLoadImageFromUrl:
    def test_load_from_http(self) -> None:
        mock_response = MagicMock()
        mock_response.content = (
            Path(__file__).parent.joinpath("images", "shoe_receipt.webp").read_bytes()
        )
        mock_response.headers = {"Content-Type": "image/webp"}

        with patch("pop_validation.imaging.loader.requests.get", return_value=mock_response):
            result = load_image("https://example.com/receipt.webp")
            assert isinstance(result, LoadedImage)
            assert result.mime_type == "image/webp"
            assert len(result.data) > 0

    def test_http_error_raises(self) -> None:
        with patch("pop_validation.imaging.loader.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            with pytest.raises(Exception, match="Connection refused"):
                load_image("https://example.com/bad.jpg")


class TestLoadImageValidation:
    def test_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_image("")

    def test_whitespace_source_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_image("   ")

    def test_mime_type_detection_png(self, tmp_path: Path) -> None:
        # Create a minimal valid PNG
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        png_path = tmp_path / "test.png"
        img.save(str(png_path))

        result = load_image(str(png_path))
        assert result.mime_type == "image/png"

    def test_mime_type_detection_jpeg(self, tmp_path: Path) -> None:
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="blue")
        jpg_path = tmp_path / "test.jpg"
        img.save(str(jpg_path))

        result = load_image(str(jpg_path))
        assert result.mime_type == "image/jpeg"

    def test_pil_image_is_valid(self, shoe_receipt_path: str) -> None:
        result = load_image(shoe_receipt_path)
        width, height = result.pil_image.size
        assert width > 0
        assert height > 0
