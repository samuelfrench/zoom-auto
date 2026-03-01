"""Zoom Meeting SDK client for joining and leaving meetings.

Uses py-zoom-meeting-sdk to connect to Zoom meetings as a bot participant.
All SDK calls are lazy-imported so the module can be tested without the
native SDK library installed.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from zoom_auto.config import Settings
from zoom_auto.zoom.events import ZoomEventHandler

logger = logging.getLogger(__name__)

# Thread pool for blocking SDK operations
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zoom-sdk")


@dataclass
class MeetingInfo:
    """Information about a Zoom meeting to join."""

    meeting_id: str
    password: str = ""
    display_name: str = "Sam's Assistant"


class ZoomClient:
    """Manages Zoom Meeting SDK connection lifecycle.

    Handles authentication, joining meetings, and graceful disconnection.
    All SDK calls run in a thread pool executor so the async event loop
    is never blocked.
    """

    def __init__(
        self,
        settings: Settings,
        event_handler: ZoomEventHandler | None = None,
    ) -> None:
        self.settings = settings
        self.events = event_handler or ZoomEventHandler()
        self._connected = False
        self._meeting_info: MeetingInfo | None = None
        self._sdk_instance: object | None = None
        self._meeting_service: object | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def join(self, meeting: MeetingInfo) -> None:
        """Join a Zoom meeting using the Meeting SDK.

        Initialises the SDK, authenticates with JWT credentials, and
        joins the specified meeting.

        Args:
            meeting: Meeting ID, password, and display name.

        Raises:
            RuntimeError: If already connected or SDK init fails.
        """
        if self._connected:
            raise RuntimeError("Already connected to a meeting")

        self._meeting_info = meeting
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(_executor, self._sdk_init)
            await loop.run_in_executor(_executor, self._sdk_auth)
            await loop.run_in_executor(
                _executor, self._sdk_join, meeting
            )
            self._connected = True
            self.events.on_meeting_joined()
            logger.info(
                "Joined meeting %s as '%s'",
                meeting.meeting_id,
                meeting.display_name,
            )
        except Exception as exc:
            logger.error("Failed to join meeting: %s", exc)
            await self._cleanup()
            raise RuntimeError(
                f"Failed to join meeting: {exc}"
            ) from exc

    async def leave(self) -> None:
        """Leave the current Zoom meeting gracefully."""
        if not self._connected:
            logger.warning("leave() called but not connected")
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(_executor, self._sdk_leave)
        except Exception:
            logger.exception("Error during SDK leave")
        finally:
            await self._cleanup()
            self.events.on_meeting_left()
            logger.info("Left meeting")

    @property
    def is_connected(self) -> bool:
        """Whether currently connected to a meeting."""
        return self._connected

    @property
    def meeting_info(self) -> MeetingInfo | None:
        """Information about the current meeting, if connected."""
        return self._meeting_info

    @property
    def sdk_instance(self) -> object | None:
        """The Zoom SDK module, if initialised."""
        return self._sdk_instance

    # ------------------------------------------------------------------ #
    #  SDK internals (run in executor)                                    #
    # ------------------------------------------------------------------ #

    def _sdk_init(self) -> None:
        """Initialise the Zoom Meeting SDK (blocking)."""
        try:
            import zoom_meeting_sdk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "zoom-meeting-sdk is not installed. "
                "Install it with: pip install zoom-meeting-sdk"
            ) from exc

        sdk = zoom_meeting_sdk
        init_params = sdk.InitParam()
        init_params.strWebDomain = "https://zoom.us"
        init_params.strSupportUrl = ""
        init_params.emLanguageID = sdk.SDK_LANGUAGE_ID.CYCUSTOM

        err = sdk.InitSDK(init_params)
        if err != sdk.SDKERR_SUCCESS:
            raise RuntimeError(f"InitSDK failed: error={err}")

        self._sdk_instance = sdk
        logger.debug("Zoom SDK initialised")

    def _sdk_auth(self) -> None:
        """Authenticate with the SDK using JWT credentials (blocking)."""
        sdk = self._sdk_instance
        if sdk is None:
            raise RuntimeError("SDK not initialised")

        auth_service = sdk.CreateAuthService()
        auth_ctx = sdk.AuthContext()
        auth_ctx.jwt_token = self._generate_jwt()

        err = auth_service.SDKAuth(auth_ctx)
        if err != sdk.SDKERR_SUCCESS:
            raise RuntimeError(f"SDK auth failed: error={err}")

        logger.debug("Zoom SDK authenticated")

    def _sdk_join(self, meeting: MeetingInfo) -> None:
        """Join the meeting (blocking)."""
        sdk = self._sdk_instance
        if sdk is None:
            raise RuntimeError("SDK not initialised")

        self._meeting_service = sdk.CreateMeetingService()
        join_params = sdk.JoinParam()
        join_params.userType = sdk.SDKUserType.SDK_UT_WITHOUT_LOGIN

        normal_param = sdk.JoinParam4NormalUser()
        normal_param.meetingNumber = int(meeting.meeting_id)
        normal_param.vanityID = ""
        normal_param.userName = meeting.display_name
        normal_param.psw = meeting.password
        normal_param.isAudioOff = False
        normal_param.isVideoOff = True

        join_params.param = normal_param

        err = self._meeting_service.Join(join_params)
        if err != sdk.SDKERR_SUCCESS:
            raise RuntimeError(f"Meeting join failed: error={err}")

        logger.debug("SDK join request sent for meeting %s", meeting.meeting_id)

    def _sdk_leave(self) -> None:
        """Leave the meeting (blocking)."""
        if self._meeting_service is not None:
            sdk = self._sdk_instance
            if sdk is not None:
                self._meeting_service.Leave(sdk.END_MEETING_REASON.EndMeetingReason_LeaveMeeting)
        logger.debug("SDK leave request sent")

    def _sdk_cleanup(self) -> None:
        """Clean up SDK resources (blocking)."""
        self._meeting_service = None
        if self._sdk_instance is not None:
            try:
                self._sdk_instance.CleanUPSDK()
            except Exception:
                logger.debug("SDK cleanup error (ignored)")
            self._sdk_instance = None
        logger.debug("SDK resources cleaned up")

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _generate_jwt(self) -> str:
        """Generate a JWT token for SDK authentication.

        Returns:
            JWT string for authenticating with the Zoom Meeting SDK.
        """
        import hashlib
        import hmac
        import json
        import time
        from base64 import urlsafe_b64encode

        key = self.settings.zoom_meeting_sdk_key
        secret = self.settings.zoom_meeting_sdk_secret

        if not key or not secret:
            raise RuntimeError(
                "zoom_meeting_sdk_key and zoom_meeting_sdk_secret "
                "must be set in settings"
            )

        iat = int(time.time())
        exp = iat + 7200  # 2 hours

        header = urlsafe_b64encode(
            json.dumps(
                {"alg": "HS256", "typ": "JWT"}, separators=(",", ":")
            ).encode()
        ).rstrip(b"=")

        payload = urlsafe_b64encode(
            json.dumps(
                {
                    "appKey": key,
                    "iat": iat,
                    "exp": exp,
                    "tokenExp": exp,
                },
                separators=(",", ":"),
            ).encode()
        ).rstrip(b"=")

        signing_input = header + b"." + payload
        signature = urlsafe_b64encode(
            hmac.new(
                secret.encode(), signing_input, hashlib.sha256
            ).digest()
        ).rstrip(b"=")

        return (signing_input + b"." + signature).decode()

    async def _cleanup(self) -> None:
        """Clean up all SDK resources."""
        self._connected = False
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(_executor, self._sdk_cleanup)
        except Exception:
            logger.debug("Cleanup error (ignored)")
        self._meeting_info = None
