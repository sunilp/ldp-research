"""Protocol abstraction layer for experiment baselines.

All protocols (LDP, A2A, MCP, raw HTTP) implement the same interface
so experiments can swap them transparently.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DelegateIdentity:
    """Delegate identity — common across all protocols."""
    id: str
    name: str
    model: str
    provider: str
    capabilities: list[str] = field(default_factory=list)

    # LDP-specific (None for non-LDP protocols)
    quality_hint: float | None = None
    cost_hint: str | None = None
    latency_hint: str | None = None
    reasoning_profile: str | None = None


@dataclass
class TaskInput:
    """A task to be delegated."""
    task_id: str
    skill: str
    prompt: str
    difficulty: str = "medium"  # easy | medium | hard
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from a delegated task."""
    task_id: str
    delegate_id: str
    output: str
    success: bool = True
    error: str | None = None

    # Metrics captured during execution
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # Provenance (LDP-specific, None for other protocols)
    provenance: dict[str, Any] | None = None

    # Protocol overhead
    overhead_messages: int = 0
    overhead_tokens: int = 0


@dataclass
class RoutingDecision:
    """Which delegate was selected and why."""
    selected_delegate_id: str
    reason: str
    candidates_considered: int = 0
    routing_latency_ms: float = 0.0


class ProtocolBaseline(ABC):
    """Base class for all protocol baselines.

    Each baseline implements:
    - discover(): find available delegates
    - route(): select best delegate for a task
    - invoke(): submit task and get result
    - invoke_session(): multi-round exchange within a session
    """

    def __init__(self, name: str, delegates: list[DelegateIdentity]):
        self.name = name
        self.delegates = {d.id: d for d in delegates}

    @abstractmethod
    async def discover(self) -> list[DelegateIdentity]:
        """Discover available delegates."""
        ...

    @abstractmethod
    async def route(self, task: TaskInput) -> RoutingDecision:
        """Select the best delegate for a task."""
        ...

    @abstractmethod
    async def invoke(
        self, delegate_id: str, task: TaskInput
    ) -> TaskResult:
        """Submit a task to a specific delegate and wait for result."""
        ...

    @abstractmethod
    async def invoke_session(
        self, delegate_id: str, tasks: list[TaskInput]
    ) -> list[TaskResult]:
        """Execute multiple tasks in a session (multi-round)."""
        ...

    async def route_and_invoke(self, task: TaskInput) -> tuple[RoutingDecision, TaskResult]:
        """Route a task to the best delegate and invoke it."""
        decision = await self.route(task)
        result = await self.invoke(decision.selected_delegate_id, task)
        return decision, result
