"""Turn-taking management — cooldowns, interruption avoidance.

Manages the timing of bot responses to ensure natural turn-taking
behavior in the conversation.
"""

from __future__ import annotations

import logging
import time

from zoom_auto.config import ResponseConfig

logger = logging.getLogger(__name__)


class TurnManager:
    """Manages turn-taking behavior for the bot.

    Enforces cooldowns between responses, limits consecutive
    responses, and avoids interrupting active speakers.
    """

    def __init__(self, config: ResponseConfig) -> None:
        self.config = config
        self._last_response_time: float = 0.0
        self._consecutive_count: int = 0

    def can_speak(self) -> bool:
        """Check if the bot is allowed to speak right now.

        Considers cooldown timer and consecutive response limit.

        Returns:
            True if the bot can speak.
        """
        now = time.time()
        elapsed = now - self._last_response_time

        if elapsed < self.config.cooldown_seconds:
            return False

        if self._consecutive_count >= self.config.max_consecutive:
            return False

        return True

    def record_response(self) -> None:
        """Record that the bot just spoke."""
        self._last_response_time = time.time()
        self._consecutive_count += 1

    def record_other_speaker(self) -> None:
        """Record that someone else spoke, resetting the consecutive counter."""
        self._consecutive_count = 0

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in the cooldown period."""
        elapsed = time.time() - self._last_response_time
        remaining = self.config.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def reset(self) -> None:
        """Reset turn-taking state for a new meeting."""
        self._last_response_time = 0.0
        self._consecutive_count = 0
