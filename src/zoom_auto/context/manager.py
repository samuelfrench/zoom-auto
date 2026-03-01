"""Context manager with sliding window and summarization.

Maintains a bounded context window of the meeting conversation,
automatically summarizing older content to stay within token limits.
Integrates transcript, speaker tracking, and meeting state into a
unified prompt for the LLM response generator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from zoom_auto.config import ContextConfig
from zoom_auto.context.meeting_state import MeetingState
from zoom_auto.context.speaker_tracker import SpeakerTracker
from zoom_auto.context.transcript import TranscriptAccumulator
from zoom_auto.llm.base import LLMMessage, LLMProvider, LLMRole

logger = logging.getLogger(__name__)

# Approximate characters per token for estimation
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using simple character heuristic.

    Uses ~4 characters per token as a rough approximation.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class ContextWindow:
    """The current context window for LLM consumption.

    Attributes:
        summary: Summary of earlier conversation (before the verbatim window).
        recent_transcript: Recent transcript lines within the verbatim window.
        meeting_context: Key meeting state (agenda, decisions, etc.).
        total_tokens_estimate: Estimated total token count.
    """

    summary: str = ""
    recent_transcript: list[str] = field(default_factory=list)
    meeting_context: str = ""
    total_tokens_estimate: int = 0


class ContextManager:
    """Manages the sliding context window for LLM prompts.

    Keeps the most recent conversation in full detail (verbatim window)
    while summarizing older content to stay within token limits.
    Integrates transcript, speaker tracking, and meeting state.

    Args:
        config: Context management configuration.
        llm: LLM provider for summarization. Optional for testing.
        transcript: Transcript accumulator. Created if not provided.
        speaker_tracker: Speaker tracker. Created if not provided.
        meeting_state: Meeting state tracker. Created if not provided.
    """

    def __init__(
        self,
        config: ContextConfig | None = None,
        llm: LLMProvider | None = None,
        transcript: TranscriptAccumulator | None = None,
        speaker_tracker: SpeakerTracker | None = None,
        meeting_state: MeetingState | None = None,
    ) -> None:
        self.config = config if config is not None else ContextConfig()
        self._llm = llm
        self._transcript = (
            transcript if transcript is not None else TranscriptAccumulator()
        )
        self._speaker_tracker = (
            speaker_tracker if speaker_tracker is not None else SpeakerTracker()
        )
        self._meeting_state = (
            meeting_state if meeting_state is not None
            else MeetingState(config=self.config)
        )
        self._window = ContextWindow()

        # Summarization state
        self._summaries: list[str] = []
        self._last_summary_time: float = time.time()
        self._summarized_up_to_segment_id: int = 0

        # Lock for concurrent access protection
        self._lock = asyncio.Lock()

    @property
    def transcript(self) -> TranscriptAccumulator:
        """The transcript accumulator."""
        return self._transcript

    @property
    def speaker_tracker(self) -> SpeakerTracker:
        """The speaker tracker."""
        return self._speaker_tracker

    @property
    def meeting_state(self) -> MeetingState:
        """The meeting state tracker."""
        return self._meeting_state

    @property
    def summaries(self) -> list[str]:
        """All generated summaries (oldest first)."""
        return self._summaries.copy()

    async def add_transcript(
        self,
        speaker: str,
        text: str,
        timestamp: datetime | None = None,
        speaker_id: int | None = None,
    ) -> None:
        """Add a new transcript utterance.

        Records the transcript entry, updates speaker tracking,
        and triggers summarization if needed.

        Args:
            speaker: Display name of the speaker.
            text: What was said.
            timestamp: When it was said. Defaults to now.
            speaker_id: Zoom user ID for speaker tracking.
        """
        if not text.strip():
            return

        async with self._lock:
            # Add to transcript
            self._transcript.add(
                speaker=speaker,
                text=text,
                timestamp=timestamp,
            )

            # Update speaker tracking
            if speaker_id is not None:
                self._speaker_tracker.register_speaker(speaker_id, speaker)
                self._speaker_tracker.record_utterance(speaker_id)

            # Update participants in meeting state
            self._meeting_state.add_participant(speaker)

            # Check if we should summarize
            await self._maybe_summarize()

        logger.debug("Added transcript: %s: %s", speaker, text[:50])

    async def _maybe_summarize(self) -> bool:
        """Check if summarization is needed and perform it.

        Summarization happens when:
        - Enough time has passed since the last summary (summary_interval_seconds)
        - There are entries outside the verbatim window to summarize
        - An LLM provider is available

        Returns:
            True if summarization was performed.
        """
        now = time.time()
        elapsed = now - self._last_summary_time

        if elapsed < self.config.summary_interval_seconds:
            return False

        return await self.summarize_if_needed()

    async def summarize_if_needed(self) -> bool:
        """Summarize older content if needed.

        Takes entries outside the verbatim window that haven't been
        summarized yet and generates a summary via the LLM.

        Returns:
            True if summarization was performed.
        """
        if self._llm is None:
            logger.debug("No LLM provider, skipping summarization")
            return False

        # Find entries outside the verbatim window that haven't been summarized
        now = datetime.now()
        old_entries = self._transcript.get_before(
            self.config.verbatim_window_seconds,
            reference_time=now,
        )

        # Filter to only unsummarized entries
        unsummarized = [
            e for e in old_entries
            if e.segment_id > self._summarized_up_to_segment_id
        ]

        if not unsummarized:
            return False

        # Format the unsummarized text
        text_to_summarize = self._transcript.get_plain_text(unsummarized)
        if not text_to_summarize.strip():
            return False

        # Generate summary
        try:
            summary = await self._generate_summary(text_to_summarize)
            if summary:
                self._summaries.append(summary)
                self._summarized_up_to_segment_id = unsummarized[-1].segment_id
                self._last_summary_time = time.time()

                # Prune old summaries if history exceeds token limit
                await self._prune_summaries()

                logger.info(
                    "Summarized %d entries (up to segment %d)",
                    len(unsummarized),
                    self._summarized_up_to_segment_id,
                )
                return True
        except Exception:
            logger.exception("Failed to summarize transcript")

        return False

    async def _generate_summary(self, text: str) -> str:
        """Generate a summary of transcript text using the LLM.

        Args:
            text: The transcript text to summarize.

        Returns:
            Summary string.
        """
        if self._llm is None:
            return ""

        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content=(
                    "You are a meeting summarizer. Summarize the following "
                    "meeting transcript segment concisely, preserving key "
                    "points, decisions, action items, and important context. "
                    "Keep it brief -- 2-4 sentences."
                ),
            ),
            LLMMessage(
                role=LLMRole.USER,
                content=f"Summarize this transcript segment:\n\n{text}",
            ),
        ]

        response = await self._llm.generate(
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        )
        return response.text.strip()

    async def _prune_summaries(self) -> None:
        """Re-summarize oldest summaries when history exceeds token limit.

        Merges the oldest summaries into a single consolidated summary
        to keep total history within max_history_tokens.
        """
        total_history = "\n\n".join(self._summaries)
        total_tokens = estimate_tokens(total_history)

        if total_tokens <= self.config.max_history_tokens:
            return

        if len(self._summaries) < 2:
            return

        logger.info(
            "History exceeds %d tokens (%d), pruning summaries",
            self.config.max_history_tokens,
            total_tokens,
        )

        # Take the oldest half of summaries and re-summarize them
        mid = len(self._summaries) // 2
        old_summaries = self._summaries[:mid]
        remaining = self._summaries[mid:]

        old_text = "\n\n".join(old_summaries)

        if self._llm is not None:
            try:
                messages = [
                    LLMMessage(
                        role=LLMRole.SYSTEM,
                        content=(
                            "You are a meeting summarizer. Consolidate these "
                            "meeting summaries into a single, more concise "
                            "summary. Preserve the most important points, "
                            "decisions, and action items."
                        ),
                    ),
                    LLMMessage(
                        role=LLMRole.USER,
                        content=f"Consolidate these summaries:\n\n{old_text}",
                    ),
                ]
                response = await self._llm.generate(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.3,
                )
                consolidated = response.text.strip()
                self._summaries = [consolidated] + remaining
                logger.info(
                    "Pruned %d summaries into 1 consolidated summary",
                    len(old_summaries),
                )
            except Exception:
                logger.exception("Failed to prune summaries")

    async def get_context(self) -> ContextWindow:
        """Get the current context window for LLM consumption.

        Builds the context window from current state:
        - Summary of older conversation
        - Recent verbatim transcript
        - Meeting state

        Returns:
            ContextWindow with summary, recent transcript, and meeting state.
        """
        now = datetime.now()

        # Build summary from stored summaries
        summary = "\n\n".join(self._summaries) if self._summaries else ""

        # Get verbatim window transcript
        recent_entries = self._transcript.get_window(
            self.config.verbatim_window_seconds,
            reference_time=now,
        )
        recent_lines = []
        for e in recent_entries:
            time_str = e.timestamp.strftime("%H:%M")
            recent_lines.append(f"{e.speaker} ({time_str}): {e.text}")

        # Get meeting state
        meeting_context = self._meeting_state.format_state()

        # Estimate total tokens
        total_text = summary + "\n".join(recent_lines) + meeting_context
        total_tokens = estimate_tokens(total_text)

        self._window = ContextWindow(
            summary=summary,
            recent_transcript=recent_lines,
            meeting_context=meeting_context,
            total_tokens_estimate=total_tokens,
        )

        return self._window

    def build_prompt(
        self,
        system_prompt: str = "",
        meeting_metadata: str = "",
    ) -> list[LLMMessage]:
        """Build the full prompt for the response generator.

        Assembles: system prompt + meeting metadata + summarized history
        + recent transcript + meeting state.

        Args:
            system_prompt: The system prompt for the LLM persona.
            meeting_metadata: Additional meeting metadata.

        Returns:
            List of LLMMessage objects ready for the LLM.
        """
        # Build system message
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)
        if meeting_metadata:
            system_parts.append(f"\n## Meeting Info\n{meeting_metadata}")

        default_system = "You are an AI meeting assistant."
        system_content = "\n".join(system_parts) if system_parts else default_system

        # Build user context message
        context_parts = []

        # Summarized history
        if self._window.summary:
            context_parts.append(
                f"## Earlier Discussion (Summary)\n{self._window.summary}"
            )

        # Recent verbatim transcript
        if self._window.recent_transcript:
            transcript_text = "\n".join(self._window.recent_transcript)
            context_parts.append(
                f"## Recent Conversation\n{transcript_text}"
            )

        # Meeting state
        if self._window.meeting_context:
            context_parts.append(
                f"## Meeting State\n{self._window.meeting_context}"
            )

        default_context = "Meeting has just started."
        context_content = "\n\n".join(context_parts) if context_parts else default_context

        return [
            LLMMessage(role=LLMRole.SYSTEM, content=system_content),
            LLMMessage(role=LLMRole.USER, content=context_content),
        ]

    async def reset(self) -> None:
        """Reset all context for a new meeting."""
        self._window = ContextWindow()
        self._summaries.clear()
        self._last_summary_time = time.time()
        self._summarized_up_to_segment_id = 0
        self._transcript.clear()
        self._speaker_tracker.reset()
        await self._meeting_state.reset()
        logger.info("Context manager reset for new meeting")
