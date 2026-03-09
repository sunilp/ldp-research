"""Smoke test — verify the experiment pipeline end-to-end.

Runs 2 tasks through each baseline against local Ollama,
then judges them with Gemini Flash (or local Ollama).
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env file if present
_env_path = project_root / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if value and key not in os.environ:
                    os.environ[key] = value

from baselines.llm_client import call_llm
from baselines.protocol import DelegateIdentity, TaskInput
from baselines.ldp_baseline import LdpBaseline
from baselines.a2a_baseline import A2aBaseline, RandomBaseline
from experiments.evaluation.judge import judge_output


async def test_ollama_connection():
    """Step 1: Verify Ollama is reachable."""
    print("1. Testing Ollama connection...")
    try:
        resp = await call_llm(
            model="llama3.2:3b",
            provider="ollama",
            system="You are a helpful assistant.",
            prompt="Say 'hello' in one word.",
            max_tokens=32,
        )
        print(f"   OK — got {resp.output_tokens} tokens in {resp.latency_ms:.0f}ms")
        print(f"   Response: {resp.content[:100]}")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


async def test_baselines():
    """Step 2: Run a task through each baseline."""
    print("\n2. Testing baselines...")
    delegates = [
        DelegateIdentity(
            id="qwen3-8b", name="qwen3-8b", model="qwen3:8b",
            provider="ollama", capabilities=["reasoning", "code", "analysis"],
            quality_hint=0.85, cost_hint="high", reasoning_profile="deep-analytical",
        ),
        DelegateIdentity(
            id="llama-3b", name="llama-3b", model="llama3.2:3b",
            provider="ollama", capabilities=["classification", "extraction"],
            quality_hint=0.55, cost_hint="low", reasoning_profile="fast-practical",
        ),
    ]

    task_easy = TaskInput(
        task_id="smoke-easy-001", skill="classification",
        prompt="Classify the sentiment of: 'This product is amazing, I love it!'",
        difficulty="easy", domain="classification",
    )
    task_hard = TaskInput(
        task_id="smoke-hard-001", skill="reasoning",
        prompt="Explain why quicksort has O(n log n) average case but O(n²) worst case. Be concise.",
        difficulty="hard", domain="reasoning",
    )

    baselines = {
        "ldp": LdpBaseline(delegates),
        "a2a": A2aBaseline(delegates),
        "random": RandomBaseline(delegates),
    }

    for name, baseline in baselines.items():
        task = task_easy if name == "random" else task_hard
        try:
            decision, result = await baseline.route_and_invoke(task)
            print(f"   {name:8s} → delegate={decision.selected_delegate_id}, "
                  f"tokens={result.total_tokens}, latency={result.latency_ms:.0f}ms")
            print(f"            output: {result.output[:120]}...")
        except Exception as e:
            print(f"   {name:8s} → FAILED: {e}")
            return False

    return True


async def test_judge():
    """Step 3: Test LLM-as-judge (try Gemini Flash, fall back to local)."""
    has_google = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    providers = []
    if has_google:
        providers.append(("Gemini Flash", "gemini-2.0-flash", "google"))
    providers.append(("local Ollama", "qwen3:8b", "ollama"))

    for label, judge_model, judge_provider in providers:
        print(f"\n3. Testing judge ({label})...")
        try:
            score = await judge_output(
                task_prompt="Explain why the sky is blue.",
                output="The sky appears blue due to Rayleigh scattering. Shorter blue wavelengths of sunlight scatter more than longer red wavelengths when they hit molecules in the atmosphere.",
                judge_model=judge_model,
                judge_provider=judge_provider,
            )
            print(f"   OK — quality={score.quality}, correctness={score.correctness}, "
                  f"completeness={score.completeness}, overall={score.overall:.1f}")
            print(f"   Reasoning: {score.reasoning[:150]}")
            return True
        except Exception as e:
            print(f"   {label} failed: {e}")
            if judge_provider != "ollama":
                print(f"   Falling back to local Ollama...")
                continue
            return False
    return False


async def main():
    print("=" * 50)
    print("LDP Research — Smoke Test")
    print("=" * 50)

    ok1 = await test_ollama_connection()
    if not ok1:
        print("\nOllama not reachable. Start it with: ollama serve")
        return

    ok2 = await test_baselines()
    ok3 = await test_judge()

    print("\n" + "=" * 50)
    if ok1 and ok2 and ok3:
        print("All checks passed. Ready to run experiments.")
        print("\nRun experiments with:")
        print("  python -m experiments.runners.main --config experiments/configs/local.yaml")
    else:
        print("Some checks failed — see above.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
