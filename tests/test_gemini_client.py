"""Tests for GeminiClient — thin Gemini API wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from pop_validation.config import Settings
from pop_validation.gemini_client import GeminiClient


class TestGeminiClient:
    @patch("pop_validation.gemini_client.genai")
    def test_configures_api_key(self, mock_genai: MagicMock, settings: Settings) -> None:
        GeminiClient(settings)
        mock_genai.configure.assert_called_once_with(api_key=settings.gemini_api_key)

    @patch("pop_validation.gemini_client.genai")
    def test_creates_model_with_settings(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        GeminiClient(settings)
        mock_genai.GenerativeModel.assert_called_once()
        call_kwargs = mock_genai.GenerativeModel.call_args
        assert call_kwargs[1]["model_name"] == settings.gemini_model_name

    @patch("pop_validation.gemini_client.genai")
    def test_call_returns_parsed_json(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        expected = {"is_receipt": True, "quality": "high"}
        mock_model.generate_content.return_value = MagicMock(
            text=json.dumps(expected),
        )

        client = GeminiClient(settings)
        result = client.call(
            prompt="test prompt",
            image_data=b"fake-image-bytes",
            mime_type="image/jpeg",
            call_label="TEST CALL",
            image_index=0,
        )

        assert result == expected
        mock_model.generate_content.assert_called_once()

    @patch("pop_validation.gemini_client.genai")
    def test_call_sends_image_part(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = MagicMock(text="{}")

        client = GeminiClient(settings)
        client.call(
            prompt="some prompt",
            image_data=b"\x89PNG",
            mime_type="image/png",
        )

        call_args = mock_model.generate_content.call_args[0][0]
        assert call_args[0] == "some prompt"
        assert call_args[1] == {"mime_type": "image/png", "data": b"\x89PNG"}

    @patch("pop_validation.gemini_client.genai")
    def test_call_propagates_api_error(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        client = GeminiClient(settings)

        import pytest

        with pytest.raises(RuntimeError, match="503"):
            client.call(
                prompt="test",
                image_data=b"fake",
                mime_type="image/jpeg",
            )

    @patch("pop_validation.gemini_client.genai")
    def test_call_propagates_json_parse_error(
        self, mock_genai: MagicMock, settings: Settings,
    ) -> None:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value = MagicMock(text="not json")

        client = GeminiClient(settings)

        import pytest

        with pytest.raises(json.JSONDecodeError):
            client.call(
                prompt="test",
                image_data=b"fake",
                mime_type="image/jpeg",
            )
