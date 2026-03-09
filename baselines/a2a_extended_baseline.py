"""A2A + custom extensions baseline.

Tests: "Can you just add metadata to A2A instead of building LDP?"
Extends A2A Agent Card with custom fields to approximate LDP identity.
This is the critical baseline — if it performs as well as LDP, the
paper's contribution shifts.
"""

from __future__ import annotations

import time

from .llm_client import call_llm
from .protocol import (
    DelegateIdentity,
    ProtocolBaseline,
    RoutingDecision,
    TaskInput,
    TaskResult,
)


class A2aExtendedBaseline(ProtocolBaseline):
    """A2A with custom metadata extensions.

    Has access to the same metadata as LDP (quality, cost, reasoning profile)
    but without LDP's session management, payload negotiation, provenance,
    or trust domains.
    """

    def __init__(self, delegates: list[DelegateIdentity]):
        super().__init__("a2a_extended", delegates)

    async def discover(self) -> list[DelegateIdentity]:
        # Extended A2A exposes metadata via custom Agent Card fields
        return list(self.delegates.values())

    async def route(self, task: TaskInput) -> RoutingDecision:
        """Route using extended A2A metadata.

        Has access to the same identity metadata as LDP.
        This isolates the routing advantage: if A2A+extensions matches
        LDP on routing, the value is in sessions/provenance/trust, not identity.
        """
        start = time.monotonic()
        candidates = list(self.delegates.values())

        # Same scoring logic as LDP — testing identity metadata value
        scores: list[tuple[str, float]] = []
        for d in candidates:
            score = self._score_delegate(d, task)
            scores.append((d.id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id = scores[0][0]
        routing_ms = (time.monotonic() - start) * 1000

        return RoutingDecision(
            selected_delegate_id=best_id,
            reason=f"A2A+ext metadata routing (same algorithm as LDP)",
            candidates_considered=len(candidates),
            routing_latency_ms=routing_ms,
        )

    def _score_delegate(self, d: DelegateIdentity, task: TaskInput) -> float:
        """Same scoring as LDP — isolating routing from protocol features."""
        score = 0.0

        task_domain = task.domain
        if task_domain in d.capabilities or "reasoning" in d.capabilities:
            score += 2.0
        if any(cap in d.capabilities for cap in ["analysis", "code", "math"]):
            score += 1.0

        q = d.quality_hint or 0.5
        if task.difficulty == "hard":
            score += q * 5.0
        elif task.difficulty == "medium":
            score += q * 3.0
        else:
            score += (1.0 - q) * 2.0 + 1.0

        if task.difficulty == "easy" and d.cost_hint == "low":
            score += 2.0

        if d.reasoning_profile:
            if task.difficulty == "hard" and "analytical" in d.reasoning_profile:
                score += 2.0
            if task.difficulty == "easy" and "fast" in d.reasoning_profile:
                score += 1.5

        return score

    async def invoke(self, delegate_id: str, task: TaskInput) -> TaskResult:
        """Invoke via A2A (no sessions, no provenance)."""
        delegate = self.delegates[delegate_id]

        system = (
            f"You are an AI assistant. "
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
            provenance=None,  # A2A has no structured provenance
            overhead_messages=1,
            overhead_tokens=0,
        )

    async def invoke_session(
        self, delegate_id: str, tasks: list[TaskInput]
    ) -> list[TaskResult]:
        """No sessions — same as standard A2A."""
        results = []
        context = ""
        for task in tasks:
            augmented_prompt = task.prompt
            if context:
                augmented_prompt = (
                    f"Previous context:\n{context}\n\nNew task:\n{task.prompt}"
                )

            augmented_task = TaskInput(
                task_id=task.task_id,
                skill=task.skill,
                prompt=augmented_prompt,
                difficulty=task.difficulty,
                domain=task.domain,
                metadata=task.metadata,
            )
            result = await self.invoke(delegate_id, augmented_task)
            result.overhead_tokens += len(context.split()) * 2
            result.overhead_messages = 1
            results.append(result)
            context += f"\n[Round {len(results)}]: {result.output[:500]}"
        return results
