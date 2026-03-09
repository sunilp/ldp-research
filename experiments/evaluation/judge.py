"""LLM-as-judge evaluation for experiment results.

Uses a judge model to score task outputs on quality, correctness,
and other dimensions. Also computes automated metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from baselines.llm_client import call_llm


@dataclass
class JudgeScore:
    """Score from an LLM judge evaluation."""
    quality: int          # 1-10
    correctness: int      # 1-10
    completeness: int     # 1-10
    reasoning: str        # Judge's reasoning
    overall: float        # Weighted average

    @property
    def as_dict(self) -> dict:
        return {
            "quality": self.quality,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "overall": self.overall,
            "reasoning": self.reasoning,
        }


JUDGE_SYSTEM = """You are an expert evaluator. Score the following task response on three dimensions:

1. **Quality** (1-10): How well-written, clear, and useful is the response?
2. **Correctness** (1-10): Is the response factually accurate and logically sound?
3. **Completeness** (1-10): Does it fully address what was asked?

Respond in JSON format:
{"quality": N, "correctness": N, "completeness": N, "reasoning": "brief explanation"}"""


async def judge_output(
    task_prompt: str,
    output: str,
    judge_model: str = "gemini-2.0-flash",
    judge_provider: str = "google",
    reference_answer: str | None = None,
) -> JudgeScore:
    """Score a task output using an LLM judge.

    Args:
        task_prompt: The original task prompt.
        output: The delegate's output to evaluate.
        judge_model: Model to use for judging.
        judge_provider: Provider for the judge model.
        reference_answer: Optional reference answer for comparison.
    """
    prompt_parts = [
        f"## Task\n{task_prompt}\n",
        f"## Response to Evaluate\n{output}\n",
    ]
    if reference_answer:
        prompt_parts.append(f"## Reference Answer\n{reference_answer}\n")

    prompt_parts.append("## Your Evaluation (JSON)")

    response = await call_llm(
        model=judge_model,
        provider=judge_provider,
        system=JUDGE_SYSTEM,
        prompt="\n".join(prompt_parts),
        max_tokens=1024,
    )

    try:
        # Extract JSON from response
        content = response.content.strip()
        # Handle markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # Handle truncated JSON — try to extract individual fields
        try:
            scores = json.loads(content)
        except json.JSONDecodeError:
            # Response was likely truncated — extract fields with regex
            import re
            scores = {}
            for key in ("quality", "correctness", "completeness"):
                m = re.search(rf'"{key}"\s*:\s*(\d+)', content)
                if m:
                    scores[key] = int(m.group(1))
            m = re.search(r'"reasoning"\s*:\s*"([^"]*)', content)
            if m:
                scores["reasoning"] = m.group(1)
            if not scores:
                raise json.JSONDecodeError("no fields found", content, 0)
        quality = int(scores.get("quality", 5))
        correctness = int(scores.get("correctness", 5))
        completeness = int(scores.get("completeness", 5))
        reasoning = scores.get("reasoning", "")

        overall = quality * 0.3 + correctness * 0.4 + completeness * 0.3

        return JudgeScore(
            quality=quality,
            correctness=correctness,
            completeness=completeness,
            reasoning=reasoning,
            overall=overall,
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: assign middle score if parsing fails
        return JudgeScore(
            quality=5, correctness=5, completeness=5,
            reasoning=f"Parse error. Raw: {response.content[:200]}",
            overall=5.0,
        )


async def judge_routing(
    task_prompt: str,
    task_difficulty: str,
    selected_delegate: str,
    delegate_pool: list[dict],
    judge_model: str = "gemini-2.0-flash",
    judge_provider: str = "google",
) -> dict:
    """Judge whether the routing decision was optimal.

    Returns dict with:
      - optimal_delegate: who the judge thinks should have been selected
      - was_optimal: bool
      - explanation: str
    """
    delegates_desc = "\n".join(
        f"- {d['id']}: model={d.get('model', '?')}, "
        f"quality={d.get('quality_hint', '?')}, "
        f"cost={d.get('cost_hint', '?')}, "
        f"capabilities={d.get('capabilities', [])}"
        for d in delegate_pool
    )

    prompt = (
        f"## Task\n{task_prompt}\n\n"
        f"## Task Difficulty: {task_difficulty}\n\n"
        f"## Available Delegates\n{delegates_desc}\n\n"
        f"## Selected Delegate: {selected_delegate}\n\n"
        f"Which delegate would be optimal for this task? "
        f'Respond in JSON: {{"optimal_delegate": "id", "was_optimal": true/false, "explanation": "why"}}'
    )

    response = await call_llm(
        model=judge_model,
        provider=judge_provider,
        system="You are an expert at matching AI models to tasks. Be concise.",
        prompt=prompt,
        max_tokens=1024,
    )

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except (json.JSONDecodeError, KeyError):
        # Try extracting from truncated JSON
        import re
        result = {}
        m = re.search(r'"optimal_delegate"\s*:\s*"([^"]*)"', content)
        if m:
            result["optimal_delegate"] = m.group(1)
        m = re.search(r'"was_optimal"\s*:\s*(true|false)', content)
        if m:
            result["was_optimal"] = m.group(1) == "true"
        m = re.search(r'"explanation"\s*:\s*"([^"]*)', content)
        if m:
            result["explanation"] = m.group(1)
        if result:
            result.setdefault("optimal_delegate", "unknown")
            result.setdefault("was_optimal", False)
            result.setdefault("explanation", "")
            return result
        return {
            "optimal_delegate": "unknown",
            "was_optimal": False,
            "explanation": f"Parse error: {response.content[:200]}",
        }
