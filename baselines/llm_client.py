"""Unified LLM client for experiments.

Wraps multiple providers with token counting, cost tracking,
and latency measurement. All baselines use this for actual LLM calls.

Supported providers:
- ollama:   Local models via Ollama (free)
- google:   Gemini models via Google AI Studio (cheap)
- anthropic: Claude models via Anthropic API
- openai:   GPT models via OpenAI API
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

# Cost per million tokens (as of 2026-03)
# Ollama models are free (local), cost tracked as 0
COST_TABLE = {
    # Ollama (free — local inference)
    "qwen3:8b": {"input": 0.0, "output": 0.0},
    "qwen2.5-coder:7b": {"input": 0.0, "output": 0.0},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
    "llama3.2:latest": {"input": 0.0, "output": 0.0},
    "gemma2:9b": {"input": 0.0, "output": 0.0},
    "phi3:mini": {"input": 0.0, "output": 0.0},
    # Google (cheap)
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class LlmResponse:
    """Response from an LLM call."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Global callback for logging all LLM calls.
# Set by experiment runner to capture every exchange.
_llm_call_hook: Any = None


def set_llm_call_hook(hook):
    """Register a callback: hook(model, provider, system, prompt, response: LlmResponse)"""
    global _llm_call_hook
    _llm_call_hook = hook


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost in USD for a given model and token counts."""
    costs = COST_TABLE.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


async def call_llm(
    model: str,
    provider: str,
    system: str | None,
    prompt: str,
    max_tokens: int = 2048,
) -> LlmResponse:
    """Call an LLM and return a structured response with metrics.

    Supports: ollama, google, anthropic, openai.
    """
    start = time.monotonic()

    if provider == "ollama":
        resp = await _call_ollama(model, system, prompt, max_tokens, start)
    elif provider == "google":
        resp = await _call_google(model, system, prompt, max_tokens, start)
    elif provider == "anthropic":
        resp = await _call_anthropic(model, system, prompt, max_tokens, start)
    elif provider == "openai":
        resp = await _call_openai(model, system, prompt, max_tokens, start)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if _llm_call_hook:
        try:
            _llm_call_hook(model, provider, system, prompt, resp)
        except Exception:
            pass  # Never let logging break the pipeline

    return resp


async def _call_ollama(
    model: str, system: str | None, prompt: str, max_tokens: int, start: float
) -> LlmResponse:
    """Call Ollama local server via its OpenAI-compatible API."""
    import httpx

    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

    latency_ms = (time.monotonic() - start) * 1000
    content = data.get("message", {}).get("content", "")

    # Ollama provides token counts in eval_count / prompt_eval_count
    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    # Fallback: estimate from text length if counts missing
    if input_tokens == 0:
        input_tokens = len(prompt.split()) * 2  # rough estimate
    if output_tokens == 0:
        output_tokens = len(content.split()) * 2

    return LlmResponse(
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=0.0,  # Local — free
    )


async def _call_google(
    model: str, system: str | None, prompt: str, max_tokens: int, start: float
) -> LlmResponse:
    """Call Google Gemini via the REST API (no SDK dependency issues)."""
    import httpx

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:generateContent?key={api_key}"
    )

    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()

    latency_ms = (time.monotonic() - start) * 1000

    # Extract content from candidates[0].content.parts[0].text
    candidates = data.get("candidates", [])
    content = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            content = parts[0].get("text", "")

    # Token counts from usageMetadata
    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    return LlmResponse(
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=_compute_cost(model, input_tokens, output_tokens),
    )


async def _call_anthropic(
    model: str, system: str | None, prompt: str, max_tokens: int, start: float
) -> LlmResponse:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)

    latency_ms = (time.monotonic() - start) * 1000
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    content = response.content[0].text if response.content else ""

    return LlmResponse(
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=_compute_cost(model, input_tokens, output_tokens),
    )


async def _call_openai(
    model: str, system: str | None, prompt: str, max_tokens: int, start: float
) -> LlmResponse:
    """Call OpenAI API."""
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    latency_ms = (time.monotonic() - start) * 1000
    usage = response.usage
    content = response.choices[0].message.content if response.choices else ""

    return LlmResponse(
        content=content,
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=_compute_cost(model, usage.prompt_tokens, usage.completion_tokens),
    )
