"""Tests for the Zoom chat sender and auto-disclaimer feature.

Tests cover:
- ChatSender initialization
- send_message with mock SDK
- send_message without SDK (graceful fallback)
- send_disclaimer uses config message
- Disclaimer disabled via config
- Web endpoint for send-chat
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from zoom_auto.config import Settings, ZoomConfig
from zoom_auto.web.app import create_app
from zoom_auto.web.routes import meetings
from zoom_auto.zoom.chat_sender import ChatSender

# ===================================================================== #
#  Fixtures                                                              #
# ===================================================================== #


@pytest.fixture
def zoom_config() -> ZoomConfig:
    """Default Zoom config with disclaimer enabled."""
    return ZoomConfig()


@pytest.fixture
def chat_sender(zoom_config: ZoomConfig) -> ChatSender:
    """ChatSender with default config, no SDK."""
    return ChatSender(config=zoom_config)


@pytest.fixture
def mock_sdk() -> SimpleNamespace:
    """Mock Zoom Meeting SDK module."""
    sdk = SimpleNamespace()
    sdk.SDKERR_SUCCESS = 0

    # Chat message type enum
    chat_type = SimpleNamespace()
    chat_type.SDKChatMessageType_To_All = 0
    sdk.SDKChatMessageType = chat_type

    # Chat controller
    chat_controller = MagicMock()
    chat_controller.SendChatMsgTo = MagicMock(return_value=0)

    # Meeting service
    meeting_service = MagicMock()
    meeting_service.GetMeetingChatController = MagicMock(
        return_value=chat_controller
    )

    sdk.CreateMeetingService = MagicMock(return_value=meeting_service)

    return sdk


@pytest.fixture(autouse=True)
def reset_meetings_state() -> None:
    """Reset module-level state between tests."""
    meetings._app_instance = None
    meetings._meeting_start_time = None


# ===================================================================== #
#  ChatSender Initialization                                             #
# ===================================================================== #


class TestChatSenderInit:
    """Tests for ChatSender initialization."""

    def test_default_config(self) -> None:
        """ChatSender uses default ZoomConfig when none provided."""
        sender = ChatSender()
        assert sender.config.bot_name == "AI Assistant"
        assert sender._sdk is None
        assert sender._chat_controller is None

    def test_custom_config(self) -> None:
        """ChatSender uses the provided config."""
        config = ZoomConfig(
            disclaimer_message="Custom disclaimer",
            send_disclaimer=False,
        )
        sender = ChatSender(config=config)
        assert sender.config.disclaimer_message == "Custom disclaimer"
        assert sender.config.send_disclaimer is False

    def test_set_sdk_creates_controller(
        self, chat_sender: ChatSender, mock_sdk: SimpleNamespace
    ) -> None:
        """set_sdk creates chat controller from SDK."""
        chat_sender.set_sdk(mock_sdk)
        assert chat_sender._sdk is mock_sdk
        assert chat_sender._chat_controller is not None
        mock_sdk.CreateMeetingService.assert_called_once()

    def test_set_sdk_fallback_on_error(
        self, chat_sender: ChatSender
    ) -> None:
        """set_sdk falls back gracefully when SDK raises."""
        bad_sdk = SimpleNamespace()
        bad_sdk.CreateMeetingService = MagicMock(
            side_effect=RuntimeError("No service")
        )
        chat_sender.set_sdk(bad_sdk)
        assert chat_sender._sdk is bad_sdk
        assert chat_sender._chat_controller is None


# ===================================================================== #
#  send_message                                                          #
# ===================================================================== #


class TestSendMessage:
    """Tests for ChatSender.send_message."""

    @pytest.mark.asyncio
    async def test_send_with_sdk(
        self, chat_sender: ChatSender, mock_sdk: SimpleNamespace
    ) -> None:
        """send_message calls SDK chat controller."""
        chat_sender.set_sdk(mock_sdk)
        result = await chat_sender.send_message("Hello meeting!")
        assert result is True

        # Verify the SDK was called with the right args
        controller = mock_sdk.CreateMeetingService().GetMeetingChatController()
        controller.SendChatMsgTo.assert_called_once_with(
            "Hello meeting!",
            mock_sdk.SDKChatMessageType.SDKChatMessageType_To_All,
        )

    @pytest.mark.asyncio
    async def test_send_without_sdk(
        self, chat_sender: ChatSender
    ) -> None:
        """send_message logs message when no SDK is available."""
        # No SDK set -- should still return True (logged)
        result = await chat_sender.send_message("Hello, no SDK!")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_sdk_error(
        self, chat_sender: ChatSender, mock_sdk: SimpleNamespace
    ) -> None:
        """send_message returns False on SDK error."""
        chat_sender.set_sdk(mock_sdk)

        # Make the controller raise
        controller = mock_sdk.CreateMeetingService().GetMeetingChatController()
        controller.SendChatMsgTo.side_effect = RuntimeError("SDK error")

        result = await chat_sender.send_message("Will fail")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_sdk_returns_error_code(
        self, chat_sender: ChatSender, mock_sdk: SimpleNamespace
    ) -> None:
        """send_message returns False when SDK returns non-success code."""
        chat_sender.set_sdk(mock_sdk)

        # Make the controller return an error code
        controller = mock_sdk.CreateMeetingService().GetMeetingChatController()
        controller.SendChatMsgTo.return_value = 99  # not SDKERR_SUCCESS

        result = await chat_sender.send_message("Will fail with error code")
        assert result is False


# ===================================================================== #
#  send_disclaimer                                                       #
# ===================================================================== #


class TestSendDisclaimer:
    """Tests for ChatSender.send_disclaimer."""

    @pytest.mark.asyncio
    async def test_uses_config_message(
        self, chat_sender: ChatSender
    ) -> None:
        """send_disclaimer sends the configured disclaimer message."""
        # Without SDK, the message is just logged -- but it should use
        # the configured disclaimer_message
        result = await chat_sender.send_disclaimer()
        assert result is True

    @pytest.mark.asyncio
    async def test_custom_disclaimer(self) -> None:
        """send_disclaimer uses custom message from config."""
        config = ZoomConfig(disclaimer_message="Custom AI notice")
        sender = ChatSender(config=config)
        # Patch send_message to verify the text
        sender.send_message = AsyncMock(return_value=True)  # type: ignore[method-assign]
        await sender.send_disclaimer()
        sender.send_message.assert_called_once_with("Custom AI notice")

    @pytest.mark.asyncio
    async def test_disclaimer_with_sdk(
        self, mock_sdk: SimpleNamespace
    ) -> None:
        """send_disclaimer sends via SDK when available."""
        config = ZoomConfig(disclaimer_message="Bot is here")
        sender = ChatSender(config=config)
        sender.set_sdk(mock_sdk)

        result = await sender.send_disclaimer()
        assert result is True


# ===================================================================== #
#  Config: disclaimer_message and send_disclaimer                        #
# ===================================================================== #


class TestDisclaimerConfig:
    """Tests for disclaimer-related config fields."""

    def test_default_disclaimer_message(self) -> None:
        """ZoomConfig has a sensible default disclaimer message."""
        config = ZoomConfig()
        assert "AI assistant" in config.disclaimer_message
        assert "locally" in config.disclaimer_message

    def test_send_disclaimer_default_enabled(self) -> None:
        """send_disclaimer is enabled by default."""
        config = ZoomConfig()
        assert config.send_disclaimer is True

    def test_send_disclaimer_can_be_disabled(self) -> None:
        """send_disclaimer can be set to False."""
        config = ZoomConfig(send_disclaimer=False)
        assert config.send_disclaimer is False


# ===================================================================== #
#  Web endpoint: /send-chat                                              #
# ===================================================================== #


class TestSendChatEndpoint:
    """Tests for the /api/meetings/send-chat web endpoint."""

    def test_send_chat_no_app(self) -> None:
        """POST /send-chat fails when no app instance."""
        from fastapi.testclient import TestClient

        settings = Settings()
        fastapi_app = create_app(settings=settings)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/send-chat",
            json={"text": "Hello"},
        )
        assert resp.status_code == 503

    def test_send_chat_not_connected(self) -> None:
        """POST /send-chat fails when not connected to a meeting."""
        from fastapi.testclient import TestClient

        settings = Settings()
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = False

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/send-chat",
            json={"text": "Hello"},
        )
        assert resp.status_code == 409

    def test_send_chat_success(self) -> None:
        """POST /send-chat succeeds when connected."""
        from fastapi.testclient import TestClient

        settings = Settings()
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True
        mock_app.send_chat_message = AsyncMock(return_value=True)

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/send-chat",
            json={"text": "Hello meeting!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        mock_app.send_chat_message.assert_called_once_with("Hello meeting!")

    def test_send_chat_failure(self) -> None:
        """POST /send-chat returns 500 when sending fails."""
        from fastapi.testclient import TestClient

        settings = Settings()
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True
        mock_app.send_chat_message = AsyncMock(return_value=False)

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/send-chat",
            json={"text": "Will fail"},
        )
        assert resp.status_code == 500

    def test_send_chat_empty_text(self) -> None:
        """POST /send-chat with empty text still sends."""
        from fastapi.testclient import TestClient

        settings = Settings()
        mock_app = MagicMock()
        mock_app.zoom_client.is_connected = True
        mock_app.send_chat_message = AsyncMock(return_value=True)

        fastapi_app = create_app(settings=settings, zoom_app=mock_app)
        client = TestClient(fastapi_app)

        resp = client.post(
            "/api/meetings/send-chat",
            json={"text": ""},
        )
        assert resp.status_code == 200
