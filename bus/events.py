"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonaPromptUpdate:
    """Ephemeral hint from background agent to persona model (one-turn lifespan)."""

    ephemeral_hint: str = ""
    """Short steering note for the persona's next turn (e.g. tone, emphasis)."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatHistorySnapshot:
    """Chat history sent to background agent."""

    session_key: str
    messages: list[dict[str, Any]]
    current_user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)
