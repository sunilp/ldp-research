"""A2A protocol baseline — primary comparison.

Uses standard A2A Agent Card (name, skills, description) for routing.
No AI-native metadata, no sessions, no provenance.
"""

from __future__ import annotations

import random
import time

from .llm_client import call_llm
from .protocol import (
    DelegateIdentity,
    ProtocolBaseline,
    RoutingDecision,
    TaskInput,
    TaskResult,
)


class A2aBaseline(ProtocolBaseline):
    """Standard A2A baseline — skill-matching only."""

    def __init__(self, delegates: list[DelegateIdentity]):
        super().__init__("a2a", delegates)

    async def discover(self) -> list[DelegateIdentity]:
        # A2A only exposes name, skills (capabilities), description
        # No quality_hint, cost_hint, reasoning_profile
        return [
            DelegateIdentity(
                id=d.id,
                name=d.name,
                model="unknown",  # A2A doesn't expose model
                provider=d.provider,
                capabilities=d.capabilities,
            )
            for d in self.delegates.values()
        ]

    async def route(self, task: TaskInput) -> RoutingDecision:
        """Route using A2A Agent Card — skill name matching only.

        No access to quality hints, cost hints, or reasoning profiles.
        """
        start = time.monotonic()
        candidates = list(self.delegates.values())

        # Simple capability matching — all A2A gives us
        # Only match on task.domain and task.skill (no hardcoded fallback)
        matching = [
            d for d in candidates
            if any(cap in d.capabilities for cap in [task.domain, task.skill])
        ]

        if not matching:
            matching = candidates

        # A2A has no way to distinguish quality among matching delegates
        # so pick the first match (effectively arbitrary)
        selected = matching[0]
        routing_ms = (time.monotonic() - start) * 1000

        return RoutingDecision(
            selected_delegate_id=selected.id,
            reason=f"A2A skill match: first delegate with matching capability",
            candidates_considered=len(candidates),
            routing_latency_ms=routing_ms,
        )

    async def invoke(self, delegate_id: str, task: TaskInput) -> TaskResult:
        """Invoke a task via A2A — stateless, no session."""
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
            provenance=None,  # A2A has no provenance
            overhead_messages=1,  # tasks/send
            overhead_tokens=0,
        )

    async def invoke_session(
        self, delegate_id: str, tasks: list[TaskInput]
    ) -> list[TaskResult]:
        """A2A has no sessions — each task is independent.

        Context must be re-established each time by including
        prior results in the prompt.
        """
        results = []
        context = ""
        for task in tasks:
            # A2A: must re-send context each round
            augmented_prompt = task.prompt
            if context:
                augmented_prompt = (
                    f"Previous context from this conversation:\n{context}\n\n"
                    f"New task:\n{task.prompt}"
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
            # Track the context overhead
            result.overhead_tokens += len(context.split()) * 2  # Rough token estimate
            result.overhead_messages = 1  # Each round is a separate task
            results.append(result)

            # Accumulate context for next round
            context += f"\n[Round {len(results)}]: {result.output[:500]}"

        return results


class RandomBaseline(A2aBaseline):
    """Random routing baseline — lower bound."""

    def __init__(self, delegates: list[DelegateIdentity]):
        super().__init__(delegates)
        self.name = "random"

    async def route(self, task: TaskInput) -> RoutingDecision:
        """Random delegate selection."""
        start = time.monotonic()
        candidates = list(self.delegates.values())
        selected = random.choice(candidates)
        routing_ms = (time.monotonic() - start) * 1000

        return RoutingDecision(
            selected_delegate_id=selected.id,
            reason="Random selection",
            candidates_considered=len(candidates),
            routing_latency_ms=routing_ms,
        )
