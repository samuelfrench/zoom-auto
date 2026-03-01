"""Slack export analysis for persona building.

Extracts communication patterns from Slack message exports
to build a realistic persona profile. Supports the standard
Slack export JSON format (channel directories with JSON files).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SlackAnalyzer:
    """Analyzes Slack message exports for persona building.

    Extracts messaging patterns, emoji usage, reaction patterns,
    and communication style from Slack export data.
    """

    def analyze_export(
        self,
        export_dir: Path,
        user_id: str | None = None,
        user_name: str | None = None,
    ) -> list[str]:
        """Analyze a Slack export directory for a specific user.

        Slack exports have this structure:
            export_dir/
              channel-name/
                2024-01-01.json
                2024-01-02.json
              another-channel/
                ...

        Each JSON file contains a list of message objects with
        keys like "user", "text", "ts", "type", etc.

        Args:
            export_dir: Path to the extracted Slack export.
            user_id: Slack user ID to filter by (e.g., "U01ABC123").
            user_name: Username to filter by (fallback if no user_id).

        Returns:
            List of the user's messages across all channels.
        """
        if not export_dir.is_dir():
            logger.warning("Slack export dir not found: %s", export_dir)
            return []

        messages: list[str] = []

        for channel_dir in sorted(export_dir.iterdir()):
            if not channel_dir.is_dir():
                continue

            for json_file in sorted(channel_dir.glob("*.json")):
                try:
                    data = json.loads(
                        json_file.read_text(encoding="utf-8"),
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Failed to parse %s: %s", json_file, exc,
                    )
                    continue

                if not isinstance(data, list):
                    continue

                for msg in data:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") != "message":
                        continue
                    # Skip bot messages and subtypes (joins, etc.)
                    if msg.get("subtype"):
                        continue

                    # Match by user_id or user_name
                    if user_id and msg.get("user") != user_id:
                        continue
                    if (
                        not user_id
                        and user_name
                        and msg.get("user_profile", {}).get(
                            "display_name", "",
                        ).lower() != user_name.lower()
                    ):
                        continue

                    text = msg.get("text", "").strip()
                    if text:
                        messages.append(text)

        logger.info(
            "Extracted %d messages from Slack export at %s",
            len(messages), export_dir,
        )
        return messages

    def extract_patterns(
        self, messages: list[str],
    ) -> dict[str, float]:
        """Extract communication patterns from messages.

        Analyzes emoji usage, message length, thread participation,
        and response patterns.

        Args:
            messages: List of Slack messages.

        Returns:
            Dict of pattern names to frequency scores.
        """
        if not messages:
            return {}

        total = len(messages)
        patterns: dict[str, float] = {}

        # Average message length (in words)
        word_counts = [len(m.split()) for m in messages]
        patterns["avg_message_words"] = round(
            sum(word_counts) / total, 2,
        )

        # Emoji usage rate
        import re
        emoji_pattern = re.compile(r":[a-z0-9_+-]+:")
        emoji_msgs = sum(
            1 for m in messages if emoji_pattern.search(m)
        )
        patterns["emoji_usage_rate"] = round(emoji_msgs / total, 4)

        # Code block rate
        code_msgs = sum(
            1 for m in messages if "```" in m or "`" in m
        )
        patterns["code_block_rate"] = round(code_msgs / total, 4)

        # Question rate
        question_msgs = sum(1 for m in messages if "?" in m)
        patterns["question_rate"] = round(question_msgs / total, 4)

        # URL sharing rate
        url_pattern = re.compile(r"https?://")
        url_msgs = sum(
            1 for m in messages if url_pattern.search(m)
        )
        patterns["url_sharing_rate"] = round(url_msgs / total, 4)

        # Short message rate (< 10 words)
        short_msgs = sum(1 for c in word_counts if c < 10)
        patterns["short_message_rate"] = round(short_msgs / total, 4)

        return patterns
