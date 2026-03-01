"""Past meeting transcript analysis for persona building.

Extracts communication patterns from historical meeting transcripts.
Supports common transcript formats: VTT, SRT, and plain text with
speaker labels.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Pattern: "Speaker Name: text" or "Speaker Name (HH:MM): text"
_SPEAKER_LINE_RE = re.compile(
    r"^(?P<speaker>[A-Za-z][A-Za-z .'-]+?)"
    r"(?:\s*\(\d{1,2}:\d{2}(?::\d{2})?\))?\s*:\s*"
    r"(?P<text>.+)$"
)

# VTT/SRT timestamp line
_TIMESTAMP_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)

# VTT speaker tag: <v Speaker Name>text</v> or <v Speaker Name>text
_VTT_SPEAKER_RE = re.compile(
    r"<v\s+(?P<speaker>[^>]+)>(?P<text>[^<]*)"
)


@dataclass
class TranscriptExcerpt:
    """An excerpt from a meeting transcript.

    Attributes:
        speaker: Speaker name or identifier.
        text: What the speaker said.
        timestamp: When in the meeting this was said.
    """

    speaker: str
    text: str
    timestamp: str = ""


class TranscriptAnalyzer:
    """Analyzes meeting transcripts for persona building.

    Extracts speaking patterns, vocabulary, and response styles
    from historical meeting transcripts.
    """

    def analyze_file(
        self,
        path: Path,
        speaker_name: str | None = None,
    ) -> list[TranscriptExcerpt]:
        """Analyze a single transcript file.

        Supports .vtt, .srt, and plain text with speaker labels.

        Args:
            path: Path to the transcript file.
            speaker_name: If provided, only extract this speaker's
                utterances. Case-insensitive partial match.

        Returns:
            List of extracted excerpts.
        """
        if not path.exists():
            logger.warning("Transcript file not found: %s", path)
            return []

        text = path.read_text(encoding="utf-8", errors="replace")
        suffix = path.suffix.lower()

        if suffix == ".vtt":
            excerpts = self._parse_vtt(text)
        elif suffix == ".srt":
            excerpts = self._parse_srt(text)
        else:
            excerpts = self._parse_plain(text)

        if speaker_name:
            lower_name = speaker_name.lower()
            excerpts = [
                e for e in excerpts
                if lower_name in e.speaker.lower()
            ]

        return excerpts

    def analyze_directory(
        self,
        directory: Path,
        speaker_name: str,
    ) -> list[str]:
        """Analyze all transcripts in a directory for a speaker.

        Args:
            directory: Directory containing transcript files.
            speaker_name: Name of the speaker to extract.

        Returns:
            List of the speaker's utterances across all transcripts.
        """
        if not directory.is_dir():
            logger.warning("Transcript directory not found: %s", directory)
            return []

        utterances: list[str] = []
        extensions = {".vtt", ".srt", ".txt"}

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in extensions and path.is_file():
                excerpts = self.analyze_file(
                    path, speaker_name=speaker_name,
                )
                utterances.extend(e.text for e in excerpts if e.text)

        logger.info(
            "Extracted %d utterances for '%s' from %s",
            len(utterances), speaker_name, directory,
        )
        return utterances

    def _parse_vtt(self, text: str) -> list[TranscriptExcerpt]:
        """Parse WebVTT transcript format."""
        excerpts: list[TranscriptExcerpt] = []
        lines = text.split("\n")
        current_ts = ""

        for line in lines:
            line = line.strip()
            if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
                continue

            # Check for timestamp line
            ts_match = _TIMESTAMP_RE.match(line)
            if ts_match:
                current_ts = ts_match.group(1)
                continue

            # Check for VTT speaker tag
            vtt_match = _VTT_SPEAKER_RE.search(line)
            if vtt_match:
                excerpts.append(TranscriptExcerpt(
                    speaker=vtt_match.group("speaker").strip(),
                    text=vtt_match.group("text").strip(),
                    timestamp=current_ts,
                ))
                continue

            # Check for "Speaker: text" format
            speaker_match = _SPEAKER_LINE_RE.match(line)
            if speaker_match:
                excerpts.append(TranscriptExcerpt(
                    speaker=speaker_match.group("speaker").strip(),
                    text=speaker_match.group("text").strip(),
                    timestamp=current_ts,
                ))
                continue

            # Plain text line — attach to previous speaker if any
            if excerpts and line and not line.isdigit():
                excerpts[-1].text += " " + line

        return excerpts

    def _parse_srt(self, text: str) -> list[TranscriptExcerpt]:
        """Parse SRT subtitle format (with optional speaker labels)."""
        excerpts: list[TranscriptExcerpt] = []
        blocks = re.split(r"\n\s*\n", text.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 2:
                continue

            # First line: sequence number (skip)
            # Second line: timestamp
            ts_line = lines[1] if len(lines) > 1 else ""
            ts_match = _TIMESTAMP_RE.match(ts_line)
            timestamp = ts_match.group(1) if ts_match else ""

            # Remaining lines: text content
            text_start = 2 if ts_match else 1
            content = " ".join(
                ln.strip() for ln in lines[text_start:] if ln.strip()
            )

            if not content:
                continue

            # Check for speaker label
            speaker_match = _SPEAKER_LINE_RE.match(content)
            if speaker_match:
                excerpts.append(TranscriptExcerpt(
                    speaker=speaker_match.group("speaker").strip(),
                    text=speaker_match.group("text").strip(),
                    timestamp=timestamp,
                ))
            else:
                excerpts.append(TranscriptExcerpt(
                    speaker="Unknown",
                    text=content,
                    timestamp=timestamp,
                ))

        return excerpts

    def _parse_plain(self, text: str) -> list[TranscriptExcerpt]:
        """Parse plain text transcript with 'Speaker: text' lines."""
        excerpts: list[TranscriptExcerpt] = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            speaker_match = _SPEAKER_LINE_RE.match(line)
            if speaker_match:
                excerpts.append(TranscriptExcerpt(
                    speaker=speaker_match.group("speaker").strip(),
                    text=speaker_match.group("text").strip(),
                ))
            elif excerpts:
                # Continuation of previous speaker's text
                excerpts[-1].text += " " + line

        return excerpts
