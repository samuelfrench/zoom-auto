"""Meeting state tracking — agenda, action items, decisions.

Tracks structured meeting state that evolves during the conversation,
providing context for response generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

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

    Maintains agenda items, action items, decisions, and key topics
    discussed during the meeting.
    """

    def __init__(self, config: ContextConfig) -> None:
        self.config = config
        self._agenda: list[str] = []
        self._action_items: list[ActionItem] = []
        self._decisions: list[Decision] = []
        self._topics: list[str] = []

    async def add_action_item(self, description: str, assignee: str = "") -> None:
        """Add an action item identified during the meeting.

        Args:
            description: What needs to be done.
            assignee: Who is responsible (if identified).
        """
        if len(self._action_items) < self.config.max_action_items:
            self._action_items.append(
                ActionItem(description=description, assignee=assignee)
            )

    async def add_decision(self, description: str, context: str = "") -> None:
        """Record a decision made during the meeting.

        Args:
            description: What was decided.
            context: Context around the decision.
        """
        self._decisions.append(Decision(description=description, context=context))

    async def add_topic(self, topic: str) -> None:
        """Add a topic discussed in the meeting.

        Args:
            topic: The topic description.
        """
        if topic not in self._topics:
            self._topics.append(topic)

    def format_state(self) -> str:
        """Format the current meeting state as text for LLM context.

        Returns:
            Formatted meeting state string.
        """
        parts = []
        if self._agenda:
            parts.append("Agenda: " + ", ".join(self._agenda))
        if self._topics:
            parts.append("Topics discussed: " + ", ".join(self._topics))
        if self._decisions:
            parts.append(
                "Decisions: "
                + "; ".join(d.description for d in self._decisions)
            )
        if self._action_items:
            parts.append(
                "Action items: "
                + "; ".join(
                    f"{a.description} ({a.assignee})" if a.assignee else a.description
                    for a in self._action_items
                )
            )
        return "\n".join(parts)

    @property
    def action_items(self) -> list[ActionItem]:
        """All recorded action items."""
        return self._action_items.copy()

    @property
    def decisions(self) -> list[Decision]:
        """All recorded decisions."""
        return self._decisions.copy()

    async def reset(self) -> None:
        """Reset meeting state for a new meeting."""
        self._agenda.clear()
        self._action_items.clear()
        self._decisions.clear()
        self._topics.clear()
