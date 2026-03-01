"""Tests for the Zoom SDK integration layer.

All tests mock the Zoom Meeting SDK so they can run without the
native library installed.  Tests cover:

- ZoomEventHandler: event registration, emission, SDK callback handlers,
  participant tracking, error handling in callbacks
- ZoomClient: join/leave lifecycle, SDK initialisation, error handling,
  JWT generation, cleanup
- AudioCapture: start/stop, on_audio_frame callback, resampling,
  async frame iteration, back-pressure / queue overflow, speaker names
- AudioSender: start/stop, send_frame, send_audio chunking, resampling,
  send loop pacing, is_sending property
- resample_pcm: identity, downsampling, upsampling, empty input
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zoom_auto.config import Settings, ZoomConfig
from zoom_auto.zoom.audio_capture import AudioCapture, AudioFrame, resample_pcm
from zoom_auto.zoom.audio_sender import AudioSender
from zoom_auto.zoom.client import MeetingInfo, ZoomClient
from zoom_auto.zoom.events import (
    MeetingEvent,
    ParticipantInfo,
    ZoomEventHandler,
)

# ===================================================================== #
#  Fixtures                                                              #
# ===================================================================== #


@pytest.fixture
def event_handler() -> ZoomEventHandler:
    """Fresh ZoomEventHandler."""
    return ZoomEventHandler()


@pytest.fixture
def settings() -> Settings:
    """Settings with dummy SDK credentials."""
    return Settings(
        zoom_meeting_sdk_key="test_key",
        zoom_meeting_sdk_secret="test_secret",
    )


@pytest.fixture
def meeting_info() -> MeetingInfo:
    """Sample meeting info."""
    return MeetingInfo(
        meeting_id="1234567890",
        password="abc123",
        display_name="Sam's Assistant",
    )


@pytest.fixture
def zoom_config() -> ZoomConfig:
    """Default Zoom config."""
    return ZoomConfig()


@pytest.fixture
def audio_capture(zoom_config: ZoomConfig) -> AudioCapture:
    """AudioCapture with default config."""
    return AudioCapture(config=zoom_config, target_sample_rate=16000)


@pytest.fixture
def audio_sender(zoom_config: ZoomConfig) -> AudioSender:
    """AudioSender with default config."""
    return AudioSender(config=zoom_config)


def _make_pcm(num_samples: int, sample_rate: int = 16000) -> bytes:
    """Helper: generate silent 16-bit PCM bytes."""
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def _make_sine_pcm(
    duration_s: float, sample_rate: int = 16000, freq: float = 440.0
) -> bytes:
    """Helper: generate a sine wave as 16-bit PCM."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return audio.tobytes()


# ===================================================================== #
#  ZoomEventHandler tests                                                #
# ===================================================================== #


class TestZoomEventHandler:
    """Tests for ZoomEventHandler event system."""

    def test_on_and_emit(self, event_handler: ZoomEventHandler) -> None:
        """Registering a callback and emitting should invoke it."""
        received: list[tuple] = []
        event_handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda evt, data: received.append((evt, data)),
        )
        event_handler.emit(MeetingEvent.MEETING_JOINED, {"key": "val"})
        assert len(received) == 1
        assert received[0] == (MeetingEvent.MEETING_JOINED, {"key": "val"})

    def test_emit_no_listeners(self, event_handler: ZoomEventHandler) -> None:
        """Emitting with no listeners should not raise."""
        event_handler.emit(MeetingEvent.MEETING_ENDED)

    def test_multiple_callbacks(self, event_handler: ZoomEventHandler) -> None:
        """Multiple callbacks on the same event should all fire."""
        calls = []
        event_handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda e, d: calls.append("a"),
        )
        event_handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda e, d: calls.append("b"),
        )
        event_handler.emit(MeetingEvent.MEETING_JOINED)
        assert calls == ["a", "b"]

    def test_off_removes_callback(self, event_handler: ZoomEventHandler) -> None:
        """off() should remove a previously registered callback."""
        calls = []
        cb = lambda e, d: calls.append(1)  # noqa: E731
        event_handler.on(MeetingEvent.MEETING_JOINED, cb)
        event_handler.off(MeetingEvent.MEETING_JOINED, cb)
        event_handler.emit(MeetingEvent.MEETING_JOINED)
        assert calls == []

    def test_off_nonexistent(self, event_handler: ZoomEventHandler) -> None:
        """off() with an unregistered callback should not raise."""
        event_handler.off(
            MeetingEvent.MEETING_JOINED, lambda e, d: None
        )

    def test_emit_default_data(self, event_handler: ZoomEventHandler) -> None:
        """Emitting without data should pass an empty dict."""
        received: list[dict] = []
        event_handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda e, d: received.append(d),
        )
        event_handler.emit(MeetingEvent.MEETING_JOINED)
        assert received == [{}]

    def test_callback_exception_does_not_propagate(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """A failing callback should not prevent other callbacks."""
        calls = []

        def bad_cb(e: MeetingEvent, d: dict) -> None:
            raise ValueError("boom")

        event_handler.on(MeetingEvent.MEETING_JOINED, bad_cb)
        event_handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda e, d: calls.append("ok"),
        )
        event_handler.emit(MeetingEvent.MEETING_JOINED)
        assert calls == ["ok"]

    def test_participants_initially_empty(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """No participants initially."""
        assert event_handler.participants == {}

    def test_participants_returns_copy(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """participants property should return a copy."""
        event_handler.on_participant_joined(1, "Alice")
        p = event_handler.participants
        p.pop(1)
        # Original should be unaffected
        assert 1 in event_handler.participants


class TestZoomEventHandlerSDKCallbacks:
    """Tests for SDK callback handler methods."""

    def test_on_meeting_joined(self, event_handler: ZoomEventHandler) -> None:
        """on_meeting_joined should emit MEETING_JOINED."""
        received = []
        event_handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda e, d: received.append(e),
        )
        event_handler.on_meeting_joined()
        assert received == [MeetingEvent.MEETING_JOINED]

    def test_on_meeting_left_clears_participants(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_meeting_left should clear participant list."""
        event_handler.on_participant_joined(1, "Alice")
        event_handler.on_meeting_left()
        assert event_handler.participants == {}

    def test_on_meeting_ended_clears_participants(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_meeting_ended should clear participant list."""
        event_handler.on_participant_joined(1, "Alice")
        event_handler.on_meeting_ended()
        assert event_handler.participants == {}

    def test_on_participant_joined_adds(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_participant_joined should add to participants dict."""
        event_handler.on_participant_joined(42, "Bob", is_host=True)
        p = event_handler.participants
        assert 42 in p
        assert p[42].display_name == "Bob"
        assert p[42].is_host is True

    def test_on_participant_joined_emits_event(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_participant_joined should emit PARTICIPANT_JOINED."""
        received: list[dict] = []
        event_handler.on(
            MeetingEvent.PARTICIPANT_JOINED,
            lambda e, d: received.append(d),
        )
        event_handler.on_participant_joined(1, "Alice")
        assert len(received) == 1
        assert isinstance(received[0]["participant"], ParticipantInfo)

    def test_on_participant_left_removes(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_participant_left should remove from participants."""
        event_handler.on_participant_joined(1, "Alice")
        event_handler.on_participant_left(1)
        assert event_handler.participants == {}

    def test_on_participant_left_unknown_user(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_participant_left for unknown user should not raise."""
        event_handler.on_participant_left(999)  # no error

    def test_on_audio_started_unmutes(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_audio_started should set is_muted = False."""
        event_handler.on_participant_joined(1, "Alice")
        event_handler._participants[1].is_muted = True
        event_handler.on_audio_started(1)
        assert event_handler._participants[1].is_muted is False

    def test_on_audio_stopped_mutes(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_audio_stopped should set is_muted = True."""
        event_handler.on_participant_joined(1, "Alice")
        event_handler.on_audio_stopped(1)
        assert event_handler._participants[1].is_muted is True

    def test_on_speaker_changed(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_speaker_changed should emit SPEAKER_CHANGED with user_id."""
        received: list[dict] = []
        event_handler.on(
            MeetingEvent.SPEAKER_CHANGED,
            lambda e, d: received.append(d),
        )
        event_handler.on_participant_joined(5, "Charlie")
        event_handler.on_speaker_changed(5)
        assert received[0]["user_id"] == 5

    def test_on_connection_error(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """on_connection_error should emit CONNECTION_ERROR."""
        received: list[dict] = []
        event_handler.on(
            MeetingEvent.CONNECTION_ERROR,
            lambda e, d: received.append(d),
        )
        event_handler.on_connection_error(500, "timeout")
        assert received[0]["error_code"] == 500
        assert received[0]["message"] == "timeout"

    def test_get_participant_found(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """get_participant should return info for known user."""
        event_handler.on_participant_joined(1, "Alice")
        info = event_handler.get_participant(1)
        assert info is not None
        assert info.display_name == "Alice"

    def test_get_participant_not_found(
        self, event_handler: ZoomEventHandler
    ) -> None:
        """get_participant should return None for unknown user."""
        assert event_handler.get_participant(999) is None


# ===================================================================== #
#  ZoomClient tests                                                      #
# ===================================================================== #


def _mock_sdk() -> SimpleNamespace:
    """Create a mock zoom_meeting_sdk module."""
    sdk = SimpleNamespace()
    sdk.SDKERR_SUCCESS = 0
    sdk.SDK_LANGUAGE_ID = SimpleNamespace(CYCUSTOM=0)

    sdk.InitParam = MagicMock
    sdk.InitSDK = MagicMock(return_value=0)
    sdk.CleanUPSDK = MagicMock()

    sdk.CreateAuthService = MagicMock(
        return_value=MagicMock(SDKAuth=MagicMock(return_value=0))
    )
    sdk.AuthContext = MagicMock
    sdk.CreateMeetingService = MagicMock(
        return_value=MagicMock(
            Join=MagicMock(return_value=0),
            Leave=MagicMock(),
        )
    )
    sdk.JoinParam = MagicMock
    sdk.JoinParam4NormalUser = MagicMock
    sdk.SDKUserType = SimpleNamespace(SDK_UT_WITHOUT_LOGIN=0)
    sdk.END_MEETING_REASON = SimpleNamespace(EndMeetingReason_LeaveMeeting=0)
    return sdk


class TestZoomClient:
    """Tests for ZoomClient lifecycle."""

    @pytest.mark.asyncio
    async def test_initial_state(self, settings: Settings) -> None:
        """Client should start disconnected with no meeting info."""
        client = ZoomClient(settings)
        assert client.is_connected is False
        assert client.meeting_info is None

    @pytest.mark.asyncio
    async def test_join_success(
        self, settings: Settings, meeting_info: MeetingInfo
    ) -> None:
        """Successful join should set connected state."""
        mock_sdk = _mock_sdk()
        client = ZoomClient(settings)

        with patch.dict("sys.modules", {"zoom_meeting_sdk": mock_sdk}):
            await client.join(meeting_info)

        assert client.is_connected is True
        assert client.meeting_info is not None
        assert client.meeting_info.meeting_id == "1234567890"

    @pytest.mark.asyncio
    async def test_join_already_connected(
        self, settings: Settings, meeting_info: MeetingInfo
    ) -> None:
        """Joining while connected should raise RuntimeError."""
        mock_sdk = _mock_sdk()
        client = ZoomClient(settings)

        with patch.dict("sys.modules", {"zoom_meeting_sdk": mock_sdk}):
            await client.join(meeting_info)
            with pytest.raises(RuntimeError, match="Already connected"):
                await client.join(meeting_info)

    @pytest.mark.asyncio
    async def test_join_emits_event(
        self, settings: Settings, meeting_info: MeetingInfo
    ) -> None:
        """Joining should emit MEETING_JOINED event."""
        mock_sdk = _mock_sdk()
        handler = ZoomEventHandler()
        received = []
        handler.on(
            MeetingEvent.MEETING_JOINED,
            lambda e, d: received.append(e),
        )
        client = ZoomClient(settings, event_handler=handler)

        with patch.dict("sys.modules", {"zoom_meeting_sdk": mock_sdk}):
            await client.join(meeting_info)

        assert MeetingEvent.MEETING_JOINED in received

    @pytest.mark.asyncio
    async def test_join_sdk_not_installed(
        self, settings: Settings, meeting_info: MeetingInfo
    ) -> None:
        """Join should raise if zoom-meeting-sdk is not importable."""
        client = ZoomClient(settings)
        # Ensure the module is not available
        with patch.dict("sys.modules", {"zoom_meeting_sdk": None}):
            with pytest.raises(RuntimeError, match="Failed to join"):
                await client.join(meeting_info)
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_leave_success(
        self, settings: Settings, meeting_info: MeetingInfo
    ) -> None:
        """Leave should disconnect and emit event."""
        mock_sdk = _mock_sdk()
        handler = ZoomEventHandler()
        received = []
        handler.on(
            MeetingEvent.MEETING_LEFT,
            lambda e, d: received.append(e),
        )
        client = ZoomClient(settings, event_handler=handler)

        with patch.dict("sys.modules", {"zoom_meeting_sdk": mock_sdk}):
            await client.join(meeting_info)
            await client.leave()

        assert client.is_connected is False
        assert MeetingEvent.MEETING_LEFT in received

    @pytest.mark.asyncio
    async def test_leave_not_connected(self, settings: Settings) -> None:
        """Leave when not connected should be a no-op."""
        client = ZoomClient(settings)
        await client.leave()  # should not raise
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_join_init_failure(
        self, settings: Settings, meeting_info: MeetingInfo
    ) -> None:
        """SDK init failure should raise and clean up."""
        mock_sdk = _mock_sdk()
        mock_sdk.InitSDK = MagicMock(return_value=99)  # non-success
        client = ZoomClient(settings)

        with patch.dict("sys.modules", {"zoom_meeting_sdk": mock_sdk}):
            with pytest.raises(RuntimeError, match="Failed to join"):
                await client.join(meeting_info)

        assert client.is_connected is False

    def test_meeting_info_dataclass(self) -> None:
        """MeetingInfo should have sensible defaults."""
        info = MeetingInfo(meeting_id="123")
        assert info.password == ""
        assert info.display_name == "Sam's Assistant"

    def test_generate_jwt(self, settings: Settings) -> None:
        """JWT generation should produce a valid three-part token."""
        client = ZoomClient(settings)
        token = client._generate_jwt()
        parts = token.split(".")
        assert len(parts) == 3

    def test_generate_jwt_no_credentials(self) -> None:
        """JWT generation without creds should raise RuntimeError."""
        settings = Settings(
            zoom_meeting_sdk_key="", zoom_meeting_sdk_secret=""
        )
        client = ZoomClient(settings)
        with pytest.raises(RuntimeError, match="must be set"):
            client._generate_jwt()


# ===================================================================== #
#  resample_pcm tests                                                    #
# ===================================================================== #


class TestResamplePCM:
    """Tests for the resample_pcm utility."""

    def test_identity(self) -> None:
        """Same source and target rate should return identical data."""
        pcm = _make_pcm(1000, 16000)
        result = resample_pcm(pcm, 16000, 16000)
        assert result == pcm

    def test_downsample_length(self) -> None:
        """Downsampling 32 kHz -> 16 kHz should halve sample count."""
        pcm = _make_pcm(3200, 32000)
        result = resample_pcm(pcm, 32000, 16000)
        result_samples = len(result) // 2
        assert result_samples == 1600

    def test_upsample_length(self) -> None:
        """Upsampling 16 kHz -> 32 kHz should double sample count."""
        pcm = _make_pcm(1600, 16000)
        result = resample_pcm(pcm, 16000, 32000)
        result_samples = len(result) // 2
        assert result_samples == 3200

    def test_empty_input(self) -> None:
        """Empty input should return empty output."""
        assert resample_pcm(b"", 16000, 32000) == b""

    def test_preserves_dtype(self) -> None:
        """Output should be valid 16-bit PCM (even byte count)."""
        pcm = _make_sine_pcm(0.1, 44100)
        result = resample_pcm(pcm, 44100, 16000)
        assert len(result) % 2 == 0
        # Should be parseable as int16
        arr = np.frombuffer(result, dtype=np.int16)
        assert arr.dtype == np.int16


# ===================================================================== #
#  AudioCapture tests                                                    #
# ===================================================================== #


class TestAudioCapture:
    """Tests for AudioCapture."""

    @pytest.mark.asyncio
    async def test_initial_state(
        self, audio_capture: AudioCapture
    ) -> None:
        """Should be inactive initially."""
        assert audio_capture.is_active is False
        assert audio_capture.queue_size == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, audio_capture: AudioCapture) -> None:
        """Start should activate, stop should deactivate."""
        await audio_capture.start()
        assert audio_capture.is_active is True
        await audio_capture.stop()
        assert audio_capture.is_active is False

    @pytest.mark.asyncio
    async def test_start_idempotent(
        self, audio_capture: AudioCapture
    ) -> None:
        """Starting twice should not raise."""
        await audio_capture.start()
        await audio_capture.start()
        assert audio_capture.is_active is True
        await audio_capture.stop()

    @pytest.mark.asyncio
    async def test_on_audio_frame_queues(
        self, audio_capture: AudioCapture
    ) -> None:
        """on_audio_frame should add a frame to the queue."""
        await audio_capture.start()
        pcm = _make_pcm(320, 16000)
        audio_capture.on_audio_frame(1, pcm, sample_rate=16000)
        assert audio_capture.queue_size == 1
        await audio_capture.stop()

    @pytest.mark.asyncio
    async def test_on_audio_frame_inactive_ignored(
        self, audio_capture: AudioCapture
    ) -> None:
        """Frames should be ignored when capture is not active."""
        pcm = _make_pcm(320, 16000)
        audio_capture.on_audio_frame(1, pcm, sample_rate=16000)
        assert audio_capture.queue_size == 0

    @pytest.mark.asyncio
    async def test_on_audio_frame_resamples(
        self, audio_capture: AudioCapture
    ) -> None:
        """Frame at 32 kHz should be resampled to 16 kHz."""
        await audio_capture.start()
        pcm_32k = _make_pcm(640, 32000)  # 640 samples at 32 kHz
        audio_capture.on_audio_frame(1, pcm_32k, sample_rate=32000)

        # Pull the frame
        frame = audio_capture._queue.get_nowait()
        assert isinstance(frame, AudioFrame)
        # Should have ~320 samples (16 kHz equivalent)
        result_samples = len(frame.pcm_data) // 2
        assert result_samples == 320
        assert frame.sample_rate == 16000
        await audio_capture.stop()

    @pytest.mark.asyncio
    async def test_frames_iteration(
        self, audio_capture: AudioCapture
    ) -> None:
        """frames() should yield queued frames then stop on sentinel."""
        await audio_capture.start()

        # Queue two frames
        pcm = _make_pcm(320, 16000)
        audio_capture.on_audio_frame(1, pcm, 16000, speaker_name="Alice")
        audio_capture.on_audio_frame(2, pcm, 16000, speaker_name="Bob")

        # Schedule stop so iteration terminates
        async def delayed_stop() -> None:
            await asyncio.sleep(0.05)
            await audio_capture.stop()

        asyncio.create_task(delayed_stop())

        collected: list[AudioFrame] = []
        async for frame in audio_capture.frames():
            collected.append(frame)

        assert len(collected) == 2
        assert collected[0].speaker_name == "Alice"
        assert collected[1].speaker_name == "Bob"

    @pytest.mark.asyncio
    async def test_speaker_name_tracking(
        self, audio_capture: AudioCapture
    ) -> None:
        """Speaker names should be remembered across frames."""
        await audio_capture.start()
        pcm = _make_pcm(320, 16000)

        audio_capture.on_audio_frame(1, pcm, 16000, speaker_name="Alice")
        # Second frame without name — should reuse cached name
        audio_capture.on_audio_frame(1, pcm, 16000)

        f1 = audio_capture._queue.get_nowait()
        f2 = audio_capture._queue.get_nowait()
        assert f1.speaker_name == "Alice"
        assert f2.speaker_name == "Alice"
        await audio_capture.stop()

    @pytest.mark.asyncio
    async def test_set_speaker_name(
        self, audio_capture: AudioCapture
    ) -> None:
        """set_speaker_name should update the name mapping."""
        audio_capture.set_speaker_name(99, "Charlie")
        await audio_capture.start()
        pcm = _make_pcm(320, 16000)
        audio_capture.on_audio_frame(99, pcm, 16000)
        f = audio_capture._queue.get_nowait()
        assert f.speaker_name == "Charlie"
        await audio_capture.stop()

    @pytest.mark.asyncio
    async def test_queue_overflow_drops_oldest(self) -> None:
        """When queue is full, oldest frame should be dropped."""
        cap = AudioCapture(max_buffer_size=2)
        await cap.start()
        pcm = _make_pcm(320, 16000)

        # Fill queue to capacity, then add one more
        cap.on_audio_frame(1, pcm, 16000, speaker_name="A")
        cap.on_audio_frame(2, pcm, 16000, speaker_name="B")
        cap.on_audio_frame(3, pcm, 16000, speaker_name="C")

        # Queue should still have 2 items
        assert cap.queue_size == 2
        # The oldest (A) should have been dropped
        f1 = cap._queue.get_nowait()
        f2 = cap._queue.get_nowait()
        names = {f1.speaker_name, f2.speaker_name}
        # B or C should be present, A was dropped
        assert "C" in names
        await cap.stop()


# ===================================================================== #
#  AudioSender tests                                                     #
# ===================================================================== #


class TestAudioSender:
    """Tests for AudioSender."""

    @pytest.mark.asyncio
    async def test_initial_state(
        self, audio_sender: AudioSender
    ) -> None:
        """Should be inactive initially."""
        assert audio_sender.is_active is False
        assert audio_sender.is_sending is False

    @pytest.mark.asyncio
    async def test_start_stop(self, audio_sender: AudioSender) -> None:
        """Start/stop should toggle active state."""
        await audio_sender.start()
        assert audio_sender.is_active is True
        await audio_sender.stop()
        assert audio_sender.is_active is False

    @pytest.mark.asyncio
    async def test_start_idempotent(
        self, audio_sender: AudioSender
    ) -> None:
        """Starting twice should not raise."""
        await audio_sender.start()
        await audio_sender.start()
        assert audio_sender.is_active is True
        await audio_sender.stop()

    @pytest.mark.asyncio
    async def test_send_frame_queues(
        self, audio_sender: AudioSender
    ) -> None:
        """send_frame should queue resampled audio."""
        await audio_sender.start()
        pcm = _make_pcm(320, 16000)
        await audio_sender.send_frame(pcm, sample_rate=16000)
        assert audio_sender.queue_size >= 1
        await audio_sender.stop()

    @pytest.mark.asyncio
    async def test_send_frame_inactive_warning(
        self, audio_sender: AudioSender
    ) -> None:
        """send_frame when inactive should not queue."""
        pcm = _make_pcm(320, 16000)
        await audio_sender.send_frame(pcm)
        assert audio_sender.queue_size == 0

    @pytest.mark.asyncio
    async def test_send_audio_chunks(self) -> None:
        """send_audio should chunk audio into SDK-sized frames."""
        sent_frames: list[bytes] = []
        sender = AudioSender(
            send_callback=lambda data: sent_frames.append(data)
        )
        await sender.start()

        # 0.1 seconds at 16 kHz = 1600 samples
        pcm = _make_pcm(1600, 16000)
        await sender.send_audio(pcm, sample_rate=16000)

        # Let the send loop drain
        await asyncio.sleep(0.5)
        await sender.stop()

        # At 32 kHz, 0.1s = 3200 samples.
        # Frame size = 640 samples (20ms at 32kHz)
        # 3200 / 640 = 5 frames
        assert len(sent_frames) == 5

    @pytest.mark.asyncio
    async def test_send_audio_empty(
        self, audio_sender: AudioSender
    ) -> None:
        """Sending empty audio should be a no-op."""
        await audio_sender.start()
        await audio_sender.send_audio(b"")
        assert audio_sender.queue_size == 0
        await audio_sender.stop()

    @pytest.mark.asyncio
    async def test_send_callback_called(self) -> None:
        """Send callback should receive each frame."""
        sent: list[bytes] = []
        sender = AudioSender(send_callback=lambda d: sent.append(d))
        await sender.start()

        pcm = _make_pcm(640, 32000)  # Exactly one 20ms frame at 32kHz
        await sender.send_frame(pcm, sample_rate=32000)

        await asyncio.sleep(0.1)
        await sender.stop()
        assert len(sent) == 1
        # Should be exactly one frame (1280 bytes = 640 samples * 2 bytes)
        assert len(sent[0]) == 1280

    @pytest.mark.asyncio
    async def test_send_callback_exception_handled(self) -> None:
        """Exception in send callback should not crash the loop."""

        def bad_callback(data: bytes) -> None:
            raise ValueError("test error")

        sender = AudioSender(send_callback=bad_callback)
        await sender.start()

        pcm = _make_pcm(640, 32000)
        await sender.send_frame(pcm, sample_rate=32000)

        await asyncio.sleep(0.1)
        # Should still be active (loop didn't crash)
        assert sender.is_active is True
        await sender.stop()

    @pytest.mark.asyncio
    async def test_send_audio_inactive(
        self, audio_sender: AudioSender
    ) -> None:
        """send_audio when inactive should not queue."""
        pcm = _make_pcm(1600, 16000)
        await audio_sender.send_audio(pcm)
        assert audio_sender.queue_size == 0

    @pytest.mark.asyncio
    async def test_is_sending_property(self) -> None:
        """is_sending should be True while frames are being sent."""
        sent: list[bytes] = []
        sender = AudioSender(send_callback=lambda d: sent.append(d))
        await sender.start()

        # Queue several frames
        for _ in range(5):
            pcm = _make_pcm(640, 32000)
            await sender.send_frame(pcm, sample_rate=32000)

        await asyncio.sleep(0.05)
        # Should be sending
        assert sender.is_sending is True

        # Wait for drain
        await asyncio.sleep(0.3)
        await sender.stop()


# ===================================================================== #
#  AudioFrame dataclass tests                                            #
# ===================================================================== #


class TestAudioFrame:
    """Tests for AudioFrame dataclass."""

    def test_audio_frame_creation(self) -> None:
        """AudioFrame should hold all fields."""
        frame = AudioFrame(
            speaker_id=1,
            speaker_name="Alice",
            pcm_data=b"\x00\x00",
            sample_rate=16000,
            timestamp_ms=1000,
        )
        assert frame.speaker_id == 1
        assert frame.speaker_name == "Alice"
        assert frame.sample_rate == 16000


# ===================================================================== #
#  MeetingEvent enum tests                                               #
# ===================================================================== #


class TestMeetingEvent:
    """Tests for MeetingEvent StrEnum."""

    def test_all_events_are_strings(self) -> None:
        """All MeetingEvent values should be strings."""
        for event in MeetingEvent:
            assert isinstance(event, str)
            assert isinstance(event.value, str)

    def test_connection_error_event_exists(self) -> None:
        """CONNECTION_ERROR should be a valid event."""
        assert MeetingEvent.CONNECTION_ERROR == "connection_error"
