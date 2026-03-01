"""Meeting state tracking -- agenda, action items, decisions.

Tracks structured meeting state that evolves during the conversation,
providing context for response generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from zoom_auto.config import ContextConfig

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """An action item identified during the meeting.

    Attributes:
        description: What needs to be done.
        assignee: Who is responsible.
        created_at: When this was identified.
    """

    description: str
    assignee: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Decision:
    """A decision made during the meeting.

    Attributes:
        description: What was decided.
        context: Context around the decision.
        created_at: When the decision was made.
    """

    description: str
    context: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class MeetingState:
    """Tracks structured meeting state.

    Maintains participants, current topic, agenda items, action items,
    decisions, and key topics discussed during the meeting.
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()
        self._participants: list[str] = []
        self._current_topic: str = ""
        self._agenda: list[str] = []
        self._action_items: list[ActionItem] = []
        self._decisions: list[Decision] = []
        self._topics: list[str] = []
        self._meeting_start_time: datetime = datetime.now()

    @property
    def meeting_start_time(self) -> datetime:
        """When the meeting started."""
        return self._meeting_start_time

    @meeting_start_time.setter
    def meeting_start_time(self, value: datetime) -> None:
        self._meeting_start_time = value

    @property
    def current_topic(self) -> str:
        """The current topic being discussed."""
        return self._current_topic

    @current_topic.setter
    def current_topic(self, value: str) -> None:
        self._current_topic = value
        # Also track in topics list if new
        if value and value not in self._topics:
            self._topics.append(value)
        logger.debug("Current topic set to: %s", value)

    @property
    def participants(self) -> list[str]:
        """List of participant names."""
        return self._participants.copy()

    def add_participant(self, name: str) -> None:
        """Add a participant to the meeting.

        Args:
            name: Participant display name.
        """
        if name not in self._participants:
            self._participants.append(name)
            logger.debug("Participant added: %s", name)

    def remove_participant(self, name: str) -> None:
        """Remove a participant from the meeting.

        Args:
            name: Participant display name.
        """
        if name in self._participants:
            self._participants.remove(name)
            logger.debug("Participant removed: %s", name)

    async def add_action_item(self, description: str, assignee: str = "") -> ActionItem | None:
        """Add an action item identified during the meeting.

        Args:
            description: What needs to be done.
            assignee: Who is responsible (if identified).

        Returns:
            The created ActionItem, or None if limit reached.
        """
        if len(self._action_items) >= self.config.max_action_items:
            logger.warning(
                "Action item limit reached (%d), ignoring: %s",
                self.config.max_action_items,
                description,
            )
            return None
        item = ActionItem(description=description, assignee=assignee)
        self._action_items.append(item)
        logger.debug("Action item added: %s (assignee: %s)", description, assignee or "unassigned")
        return item

    async def add_decision(self, description: str, context: str = "") -> Decision:
        """Record a decision made during the meeting.

        Args:
            description: What was decided.
            context: Context around the decision.

        Returns:
            The created Decision.
        """
        decision = Decision(description=description, context=context)
        self._decisions.append(decision)
        logger.debug("Decision recorded: %s", description)
        return decision

    async def add_topic(self, topic: str) -> None:
        """Add a topic discussed in the meeting.

        Args:
            topic: The topic description.
        """
        if topic not in self._topics:
            self._topics.append(topic)
            logger.debug("Topic added: %s", topic)

    def set_agenda(self, items: list[str]) -> None:
        """Set the meeting agenda.

        Args:
            items: List of agenda item descriptions.
        """
        self._agenda = list(items)
        logger.debug("Agenda set with %d items", len(items))

    def format_state(self) -> str:
        """Format the current meeting state as text for LLM context.

        Returns:
            Formatted meeting state string.
        """
        parts = []

        # Meeting metadata
        duration = datetime.now() - self._meeting_start_time
        minutes = int(duration.total_seconds() / 60)
        parts.append(f"Meeting duration: {minutes} minutes")

        if self._participants:
            parts.append("Participants: " + ", ".join(self._participants))

        if self._current_topic:
            parts.append(f"Current topic: {self._current_topic}")

        if self._agenda:
            parts.append("Agenda: " + ", ".join(self._agenda))

        if self._topics:
            parts.append("Topics discussed: " + ", ".join(self._topics))

        if self._decisions:
            decision_strs = []
            for d in self._decisions:
                s = d.description
                if d.context:
                    s += f" (context: {d.context})"
                decision_strs.append(s)
            parts.append("Decisions: " + "; ".join(decision_strs))

        if self._action_items:
            item_strs = []
            for a in self._action_items:
                s = a.description
                if a.assignee:
                    s += f" ({a.assignee})"
                item_strs.append(s)
            parts.append("Action items: " + "; ".join(item_strs))

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the meeting state to a dictionary.

        Returns:
            Dictionary representation of the meeting state.
        """
        return {
            "meeting_start_time": self._meeting_start_time.isoformat(),
            "participants": self._participants.copy(),
            "current_topic": self._current_topic,
            "agenda": self._agenda.copy(),
            "topics_discussed": self._topics.copy(),
            "decisions": [
                {
                    "description": d.description,
                    "context": d.context,
                    "created_at": d.created_at.isoformat(),
                }
                for d in self._decisions
            ],
            "action_items": [
                {
                    "description": a.description,
                    "assignee": a.assignee,
                    "created_at": a.created_at.isoformat(),
                }
                for a in self._action_items
            ],
        }

    @property
    def action_items(self) -> list[ActionItem]:
        """All recorded action items."""
        return self._action_items.copy()

    @property
    def decisions(self) -> list[Decision]:
        """All recorded decisions."""
        return self._decisions.copy()

    @property
    def topics(self) -> list[str]:
        """All discussed topics."""
        return self._topics.copy()

    async def reset(self) -> None:
        """Reset meeting state for a new meeting."""
        self._participants.clear()
        self._current_topic = ""
        self._agenda.clear()
        self._action_items.clear()
        self._decisions.clear()
        self._topics.clear()
        self._meeting_start_time = datetime.now()
        logger.debug("Meeting state reset")
