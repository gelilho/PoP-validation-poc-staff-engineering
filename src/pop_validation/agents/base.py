"""Base agent contract — Protocol + universal result type.

Every agent implements BaseAgent.execute() and returns AgentResult.
This is the only contract needed to swap in Claude Agent SDK or Google ADK later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentResult:
    """Universal message type returned by every agent.

    Attributes:
        success: Whether the agent completed without errors.
        data: Agent-specific output (LoadedImage, ImageQualityReport, etc.).
        error: Human-readable error description, or None on success.
    """

    success: bool
    data: Any = None
    error: str | None = None


class BaseAgent(Protocol):
    """Contract that every agent must satisfy.

    Agents are thin wrappers around existing domain functions.
    The orchestrator calls execute() and inspects the AgentResult.
    """

    @property
    def name(self) -> str:
        """Short identifier for logging (e.g. 'ImageLoader', 'Quality')."""
        ...

    def execute(self, **kwargs: Any) -> AgentResult:
        """Run the agent's single responsibility.

        Returns AgentResult — never raises on domain errors.
        """
        ...
