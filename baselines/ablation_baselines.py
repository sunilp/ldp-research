"""Ablation baselines — isolate routing vs prompting contributions.

Two conditions that cross the routing and prompting axes of LDP vs A2A:

1. A2aLdpPromptBaseline ("a2a_ldp_prompt"):
   - Routing: A2A's skill-match routing (capability name matching only)
   - Prompting: LDP's identity-enriched system prompt (delegate name, model, capabilities)
   -> Isolates: Does LDP's richer prompt improve output quality even when
      routing is no better than A2A?

2. LdpGenericPromptBaseline ("ldp_generic_prompt"):
   - Routing: LDP's metadata-aware routing (quality_hint, cost_hint, reasoning_profile)
   - Prompting: A2A's generic "You are an AI assistant" system prompt
   -> Isolates: Does LDP's smarter routing improve outcomes even when the
      delegate receives no identity context in its prompt?

Together with the existing LDP (both) and A2A (neither) conditions, these
form a 2x2 factorial design:

                     | A2A Routing      | LDP Routing       |
    -----------------+------------------+-------------------+
    Generic Prompt   | A2A (existing)   | ldp_generic_prompt|
    Identity Prompt  | a2a_ldp_prompt   | LDP (existing)    |
    -----------------+------------------+-------------------+
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


class A2aLdpPromptBaseline(ProtocolBaseline):
    """A2A skill-match routing + LDP identity-enriched prompting.

    Routing: identical to A2aBaseline.route() -- capability name matching only.
    Prompting: identical to LdpBaseline.invoke() -- delegate name, model, capabilities.

    This isolates the prompting contribution: if this outperforms pure A2A,
    the identity-enriched prompt is adding value independent of routing.
    """

    def __init__(self, delegates: list[DelegateIdentity]):
        super().__init__("a2a_ldp_prompt", delegates)

    async def discover(self) -> list[DelegateIdentity]:
        # A2A discovery: only name, skills, provider (no quality/cost hints)
        return [
            DelegateIdentity(
                id=d.id,
                name=d.name,
                model="unknown",
                provider=d.provider,
                capabilities=d.capabilities,
            )
            for d in self.delegates.values()
        ]

    async def route(self, task: TaskInput) -> RoutingDecision:
        """Route using A2A skill-match only (no metadata).

        Identical to A2aBaseline.route(): match on task.domain and task.skill
        against delegate capabilities, pick the first match.
        """
        start = time.monotonic()
        candidates = list(self.delegates.values())

        # Simple capability matching -- all A2A gives us
        matching = [
            d for d in candidates
            if any(cap in d.capabilities for cap in [task.domain, task.skill])
        ]

        if not matching:
            matching = candidates

        # A2A has no way to distinguish quality among matching delegates
        selected = matching[0]
        routing_ms = (time.monotonic() - start) * 1000

        return RoutingDecision(
            selected_delegate_id=selected.id,
            reason="Ablation[a2a_ldp_prompt]: A2A skill-match routing, LDP prompt",
            candidates_considered=len(candidates),
            routing_latency_ms=routing_ms,
        )

    async def invoke(self, delegate_id: str, task: TaskInput) -> TaskResult:
        """Invoke with LDP's identity-enriched system prompt.

        Identical to LdpBaseline.invoke() prompt construction:
        includes delegate name, model, and capabilities.
        """
        delegate = self.delegates[delegate_id]

        # LDP-style identity-enriched prompt
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
            provenance=None,  # No provenance in ablation
            overhead_messages=1,
            overhead_tokens=0,
        )

    async def invoke_session(
        self, delegate_id: str, tasks: list[TaskInput]
    ) -> list[TaskResult]:
        """No sessions -- same as A2A (stateless, re-send context)."""
        results = []
        context = ""
        for task in tasks:
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
            result.overhead_tokens += len(context.split()) * 2
            result.overhead_messages = 1
            results.append(result)

            context += f"\n[Round {len(results)}]: {result.output[:500]}"

        return results


class LdpGenericPromptBaseline(ProtocolBaseline):
    """LDP metadata-aware routing + A2A generic prompting.

    Routing: identical to LdpBaseline.route() -- uses quality_hint, cost_hint,
             reasoning_profile, capabilities, and task difficulty.
    Prompting: identical to A2aBaseline.invoke() -- "You are an AI assistant."

    This isolates the routing contribution: if this outperforms pure A2A,
    metadata-aware routing is adding value independent of the prompt.
    """

    def __init__(self, delegates: list[DelegateIdentity]):
        super().__init__("ldp_generic_prompt", delegates)

    async def discover(self) -> list[DelegateIdentity]:
        # Full LDP discovery (routing needs the metadata)
        return list(self.delegates.values())

    async def route(self, task: TaskInput) -> RoutingDecision:
        """Route using LDP identity metadata.

        Identical to LdpBaseline.route(): considers capabilities, quality_hint,
        cost_hint, reasoning_profile, and task difficulty.
        """
        start = time.monotonic()
        candidates = list(self.delegates.values())

        scores: list[tuple[str, float]] = []
        for d in candidates:
            score = self._score_delegate(d, task)
            scores.append((d.id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id = scores[0][0]
        routing_ms = (time.monotonic() - start) * 1000

        return RoutingDecision(
            selected_delegate_id=best_id,
            reason=f"Ablation[ldp_generic_prompt]: LDP metadata routing, generic prompt. "
                   f"quality={self.delegates[best_id].quality_hint}, "
                   f"profile={self.delegates[best_id].reasoning_profile}",
            candidates_considered=len(candidates),
            routing_latency_ms=routing_ms,
        )

    def _score_delegate(self, d: DelegateIdentity, task: TaskInput) -> float:
        """Score a delegate using LDP identity metadata.

        Identical to LdpBaseline._score_delegate().
        """
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
            score += q * 5.0
        elif task.difficulty == "medium":
            score += q * 3.0
        else:
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
        """Invoke with A2A's generic system prompt.

        Identical to A2aBaseline.invoke() prompt: no delegate identity context.
        """
        delegate = self.delegates[delegate_id]

        # A2A-style generic prompt -- no identity info
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
            provenance=None,  # No provenance in ablation
            overhead_messages=1,
            overhead_tokens=0,
        )

    async def invoke_session(
        self, delegate_id: str, tasks: list[TaskInput]
    ) -> list[TaskResult]:
        """No sessions -- same as A2A (stateless, re-send context)."""
        results = []
        context = ""
        for task in tasks:
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
            result.overhead_tokens += len(context.split()) * 2
            result.overhead_messages = 1
            results.append(result)

            context += f"\n[Round {len(results)}]: {result.output[:500]}"

        return results
