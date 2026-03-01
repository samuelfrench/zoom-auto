"""Turn-taking management -- cooldowns, interruption avoidance.

Manages the timing of bot responses to ensure natural turn-taking
behavior in the conversation.
"""

from __future__ import annotations

import logging
import random
import time

from zoom_auto.config import ResponseConfig

logger = logging.getLogger(__name__)

# Natural pause range before speaking (seconds)
_MIN_PAUSE = 0.3
_MAX_PAUSE = 0.8


class TurnManager:
    """Manages turn-taking behavior for the bot.

    Enforces cooldowns between responses, limits consecutive
    responses, tracks whether someone is currently speaking,
    and manages the bot's own speaking state.

    Args:
        config: Response engine configuration.
    """

    def __init__(self, config: ResponseConfig) -> None:
        self.config = config
        self._last_response_time: float = 0.0
        self._consecutive_count: int = 0
        self._someone_speaking: bool = False
        self._bot_speaking: bool = False

    def can_speak(self) -> bool:
        """Check if the bot is allowed to speak right now.

        Considers:
        - Cooldown timer since last bot utterance
        - Consecutive response limit
        - Whether someone else is currently speaking
        - Whether the bot is already speaking

        Returns:
            True if the bot can speak.
        """
        if self._bot_speaking:
            return False

        if self._someone_speaking:
            return False

        if self.is_cooldown_active:
            return False

        if self._consecutive_count >= self.config.max_consecutive:
            return False

        return True

    @property
    def is_cooldown_active(self) -> bool:
        """Whether the cooldown timer is still active."""
        if self._last_response_time == 0.0:
            return False
        elapsed = time.time() - self._last_response_time
        return elapsed < self.config.cooldown_seconds

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in the cooldown period."""
        if self._last_response_time == 0.0:
            return 0.0
        elapsed = time.time() - self._last_response_time
        remaining = self.config.cooldown_seconds - elapsed
        return max(0.0, remaining)

    @property
    def someone_speaking(self) -> bool:
        """Whether another participant is currently speaking."""
        return self._someone_speaking

    @property
    def bot_speaking(self) -> bool:
        """Whether the bot is currently speaking."""
        return self._bot_speaking

    def record_response(self) -> None:
        """Record that the bot just spoke.

        Updates the cooldown timer and increments the consecutive
        response counter.
        """
        self._last_response_time = time.time()
        self._consecutive_count += 1
        self._bot_speaking = False
        logger.debug(
            "Bot response recorded (consecutive: %d)",
            self._consecutive_count,
        )

    def record_other_speaker(self) -> None:
        """Record that someone else spoke, resetting the consecutive counter."""
        self._consecutive_count = 0

    def mark_bot_speaking(self) -> None:
        """Mark the bot as currently speaking (TTS output active)."""
        self._bot_speaking = True
        logger.debug("Bot started speaking")

    def mark_bot_done(self) -> None:
        """Mark the bot as finished speaking."""
        self._bot_speaking = False
        self._last_response_time = time.time()
        logger.debug("Bot finished speaking")

    def on_speech_detected(self) -> None:
        """Called when VAD detects someone else is talking."""
        was_silent = not self._someone_speaking
        self._someone_speaking = True
        if was_silent:
            logger.debug("Speech detected - someone is talking")

    def on_silence_detected(self) -> None:
        """Called when VAD detects silence (nobody talking)."""
        was_speaking = self._someone_speaking
        self._someone_speaking = False
        if was_speaking:
            logger.debug("Silence detected - nobody talking")

    def should_interrupt(self) -> bool:
        """Check if the bot should stop speaking due to interruption.

        Returns True if someone started talking while the bot is
        speaking, indicating the bot should stop its TTS output.

        Returns:
            True if the bot should stop speaking.
        """
        return self._bot_speaking and self._someone_speaking

    def get_natural_pause(self) -> float:
        """Get a random natural pause duration before speaking.

        Returns a duration between 0.3 and 0.8 seconds to simulate
        the natural delay before a person starts responding.

        Returns:
            Pause duration in seconds.
        """
        return random.uniform(_MIN_PAUSE, _MAX_PAUSE)

    def override_cooldown(self) -> None:
        """Override the cooldown (e.g., when directly addressed).

        Resets the last response time so the bot can respond
        immediately.
        """
        self._last_response_time = 0.0
        logger.debug("Cooldown overridden")

    def reset(self) -> None:
        """Reset turn-taking state for a new meeting."""
        self._last_response_time = 0.0
        self._consecutive_count = 0
        self._someone_speaking = False
        self._bot_speaking = False
        logger.debug("Turn manager reset")
