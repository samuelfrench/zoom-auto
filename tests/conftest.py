"""Shared test fixtures for zoom-auto test suite."""

import pytest


@pytest.fixture
def sample_audio_frames() -> list[bytes]:
    """Generate dummy PCM audio frames for testing."""
    import numpy as np

    frames = []
    for _ in range(10):
        # 16-bit PCM, 16kHz, 20ms frame = 320 samples
        samples = np.zeros(320, dtype=np.int16)
        frames.append(samples.tobytes())
    return frames


@pytest.fixture
def sample_transcript() -> str:
    """A sample meeting transcript for testing."""
    return (
        "Speaker 1: Let's discuss the Q2 roadmap.\n"
        "Speaker 2: I think we should focus on the API redesign.\n"
        "Speaker 1: Good point. What about the mobile app?\n"
        "Speaker 3: Mobile can wait until Q3.\n"
    )
