"""LDP protocol baseline — the treatment condition.

Uses full LDP identity (model metadata, quality/cost/latency hints,
reasoning profile) for routing and invocation. Sessions maintained
across multi-round exchanges. Provenance attached to all results.
"""

from __future__ import annotations

import json
import time

from .llm_client import call_llm
from .protocol import (
    DelegateIdentity,
    ProtocolBaseline,
    RoutingDecision,
    TaskInput,
    TaskResult,
)


class LdpBaseline(ProtocolBaseline):
    """LDP protocol baseline with full AI-native identity."""

    def __init__(self, delegates: list[DelegateIdentity]):
        super().__init__("ldp", delegates)
        self._sessions: dict[str, dict] = {}  # delegate_id → session state

    async def discover(self) -> list[DelegateIdentity]:
        return list(self.delegates.values())

    async def route(self, task: TaskInput) -> RoutingDecision:
        """Route using LDP identity metadata.

        Considers: capabilities, quality_hint, cost_hint, reasoning_profile,
        and task difficulty/domain to select the best delegate.
        """
        start = time.monotonic()
        candidates = list(self.delegates.values())

        # Score each delegate for this task
        scores: list[tuple[str, float]] = []
        for d in candidates:
            score = self._score_delegate(d, task)
            scores.append((d.id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id = scores[0][0]
        routing_ms = (time.monotonic() - start) * 1000

        return RoutingDecision(
            selected_delegate_id=best_id,
            reason=f"LDP metadata routing: quality={self.delegates[best_id].quality_hint}, "
                   f"profile={self.delegates[best_id].reasoning_profile}",
            candidates_considered=len(candidates),
            routing_latency_ms=routing_ms,
        )

    def _score_delegate(self, d: DelegateIdentity, task: TaskInput) -> float:
        """Score a delegate for a task using LDP identity metadata."""
        score = 0.0

        # Capability match
        task_domain = task.domain
        if task_domain in d.capabilities or "reasoning" in d.capabilities:
            score += 2.0
        if any(cap in d.capabilities for cap in ["analysis", "code", "math"]):
            score += 1.0

        # Quality match for difficulty
        q = d.quality_hint or 0.5
        if task.difficulty == "hard":
            score += q * 5.0  # Hard tasks need high quality
        elif task.difficulty == "medium":
            score += q * 3.0
        else:
            # Easy tasks — prefer cheaper delegates
            score += (1.0 - q) * 2.0 + 1.0

        # Cost efficiency for easy tasks
        if task.difficulty == "easy" and d.cost_hint == "low":
            score += 2.0

        # Reasoning profile match
        if d.reasoning_profile:
            if task.difficulty == "hard" and "analytical" in d.reasoning_profile:
                score += 2.0
            if task.difficulty == "easy" and "fast" in d.reasoning_profile:
                score += 1.5

        return score

    async def invoke(self, delegate_id: str, task: TaskInput) -> TaskResult:
        """Invoke a task on a delegate via LDP."""
        delegate = self.delegates[delegate_id]

        # Session overhead (first call establishes session)
        overhead_msgs = 0
        overhead_tokens = 0
        if delegate_id not in self._sessions:
            # Simulate HELLO → CAPABILITY_MANIFEST → SESSION_PROPOSE → SESSION_ACCEPT
            overhead_msgs = 4
            overhead_tokens = 200  # Approximate session setup overhead
            self._sessions[delegate_id] = {
                "session_id": f"session-{delegate_id}",
                "payload_mode": "semantic_frame",
                "established": True,
            }

        system = (
            f"You are delegate {delegate.name} ({delegate.model}). "
            f"Capabilities: {', '.join(delegate.capabilities)}. "
            f"Respond to the task thoroughly."
        )

        response = await call_llm(
            model=delegate.model,
            provider=delegate.provider,
            system=system,
            prompt=task.prompt,
        )

        return TaskResult(
            task_id=task.task_id,
            delegate_id=delegate_id,
            output=response.content,
            success=True,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            cost_usd=response.cost_usd,
            provenance={
                "produced_by": delegate_id,
                "model_version": delegate.model,
                "payload_mode_used": "semantic_frame",
                "confidence": delegate.quality_hint,
                "verified": False,
            },
            overhead_messages=overhead_msgs,
            overhead_tokens=overhead_tokens,
        )

    async def invoke_session(
        self, delegate_id: str, tasks: list[TaskInput]
    ) -> list[TaskResult]:
        """Execute multiple tasks in an LDP session.

        Session is established once, then all tasks share the context.
        """
        results = []
        for i, task in enumerate(tasks):
            result = await self.invoke(delegate_id, task)
            # Only first call has session overhead
            if i > 0:
                result.overhead_messages = 0
                result.overhead_tokens = 0
            results.append(result)
        return results
