"""Sends messages to Zoom meeting chat via the SDK.

Uses the Zoom Meeting SDK's chat controller to send text messages
to the meeting chat. All SDK calls run in a thread pool executor
to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from zoom_auto.config import ZoomConfig

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zoom-chat")


class ChatSender:
    """Sends messages to Zoom meeting chat.

    Uses the Zoom Meeting SDK's chat controller to send text messages
    to the meeting chat. All SDK calls run in a thread pool executor.
    """

    def __init__(self, config: ZoomConfig | None = None) -> None:
        self.config = config or ZoomConfig()
        self._chat_controller: object | None = None
        self._sdk: object | None = None

    def set_sdk(self, sdk: object) -> None:
        """Set the SDK instance after initialization.

        Creates the chat controller from the SDK's meeting service.

        Args:
            sdk: The Zoom Meeting SDK module instance.
        """
        self._sdk = sdk
        try:
            meeting_service = sdk.CreateMeetingService()  # type: ignore[attr-defined]
            self._chat_controller = meeting_service.GetMeetingChatController()  # type: ignore[attr-defined]
            logger.info("Chat controller initialized from SDK")
        except Exception:
            logger.warning(
                "Could not create chat controller from SDK; "
                "chat messages will be logged only"
            )
            self._chat_controller = None

    async def send_message(self, text: str) -> bool:
        """Send a text message to the meeting chat.

        If the SDK is not available, the message is logged instead.

        Args:
            text: The message text to send.

        Returns:
            True if the message was sent (or logged), False on error.
        """
        if self._chat_controller is None or self._sdk is None:
            logger.info("Chat (no SDK): %s", text)
            return True

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                _executor, self._send_blocking, text
            )
            logger.info("Chat message sent: %s", text)
            return True
        except Exception:
            logger.exception("Failed to send chat message")
            return False

    async def send_disclaimer(self) -> bool:
        """Send the configured disclaimer message to the meeting chat.

        Returns:
            True if the disclaimer was sent successfully, False otherwise.
        """
        return await self.send_message(self.config.disclaimer_message)

    def _send_blocking(self, text: str) -> None:
        """Send a chat message using the SDK (blocking call).

        Args:
            text: The message text to send.

        Raises:
            RuntimeError: If the SDK returns an error.
        """
        sdk = self._sdk
        controller = self._chat_controller

        msg_type = sdk.SDKChatMessageType.SDKChatMessageType_To_All  # type: ignore[attr-defined]
        err = controller.SendChatMsgTo(text, msg_type)  # type: ignore[attr-defined]

        if err != sdk.SDKERR_SUCCESS:  # type: ignore[attr-defined]
            raise RuntimeError(f"SendChatMsgTo failed: error={err}")
