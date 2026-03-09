"""Comprehensive experiment logging.

Records every meaningful exchange: LLM calls, routing decisions,
judge evaluations, raw inputs/outputs. Writes both a structured
JSONL log (machine-readable) and per-run detail files.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    event_type: str  # llm_call, routing, judge, task_result, experiment_start, etc.
    experiment: str
    condition: str
    task_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class ExperimentLogger:
    """Records all experiment data to structured log files.

    Produces:
    - {output_dir}/logs/{experiment}_{condition}_{timestamp}.jsonl  (all events)
    - {output_dir}/logs/runs/{task_id}.json                        (per-run detail)
    - {output_dir}/logs/summary.json                               (run metadata)
    """

    def __init__(self, output_dir: str, run_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.output_dir / "logs" / self.run_id
        self.runs_dir = self.log_dir / "runs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Main JSONL log file
        self._log_path = self.log_dir / "events.jsonl"
        self._log_file = open(self._log_path, "a")

        # Counters
        self._event_count = 0
        self._llm_call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._start_time = time.monotonic()

        # Per-task accumulators
        self._task_data: dict[str, list[dict]] = {}

    def _write_event(self, entry: LogEntry):
        """Write a log entry to the JSONL file."""
        self._log_file.write(json.dumps(asdict(entry), default=str) + "\n")
        self._log_file.flush()
        self._event_count += 1

    def log_experiment_start(self, experiment: str, config: dict):
        """Log experiment start with full config."""
        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="experiment_start",
            experiment=experiment,
            condition="",
            data={
                "run_id": self.run_id,
                "config": config,
            },
        ))

    def log_experiment_end(self, experiment: str, condition: str, summary: dict):
        """Log experiment end with summary stats."""
        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="experiment_end",
            experiment=experiment,
            condition=condition,
            data=summary,
        ))

    def log_llm_call(
        self,
        experiment: str,
        condition: str,
        task_id: str,
        *,
        purpose: str,  # "delegate", "judge_output", "judge_routing", "synthesis"
        model: str,
        provider: str,
        system: str | None,
        prompt: str,
        response_content: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
    ):
        """Log every LLM call with full input/output."""
        self._llm_call_count += 1
        self._total_tokens += input_tokens + output_tokens
        self._total_cost += cost_usd

        entry_data = {
            "purpose": purpose,
            "model": model,
            "provider": provider,
            "system": system,
            "prompt": prompt,
            "response": response_content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        }

        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="llm_call",
            experiment=experiment,
            condition=condition,
            task_id=task_id,
            data=entry_data,
        ))

        # Accumulate for per-task detail
        self._task_data.setdefault(task_id, []).append({
            "event": "llm_call", **entry_data
        })

    def log_routing(
        self,
        experiment: str,
        condition: str,
        task_id: str,
        *,
        task_prompt: str,
        task_difficulty: str,
        task_domain: str,
        candidates: list[dict],
        selected_delegate: str,
        reason: str,
        routing_latency_ms: float,
    ):
        """Log routing decision with all context."""
        entry_data = {
            "task_prompt": task_prompt,
            "task_difficulty": task_difficulty,
            "task_domain": task_domain,
            "candidates": candidates,
            "selected_delegate": selected_delegate,
            "reason": reason,
            "routing_latency_ms": routing_latency_ms,
        }

        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="routing",
            experiment=experiment,
            condition=condition,
            task_id=task_id,
            data=entry_data,
        ))

        self._task_data.setdefault(task_id, []).append({
            "event": "routing", **entry_data
        })

    def log_judge(
        self,
        experiment: str,
        condition: str,
        task_id: str,
        *,
        judge_type: str,  # "output" or "routing"
        scores: dict,
    ):
        """Log judge evaluation scores."""
        entry_data = {
            "judge_type": judge_type,
            "scores": scores,
        }

        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="judge",
            experiment=experiment,
            condition=condition,
            task_id=task_id,
            data=entry_data,
        ))

        self._task_data.setdefault(task_id, []).append({
            "event": "judge", **entry_data
        })

    def log_task_result(
        self,
        experiment: str,
        condition: str,
        task_id: str,
        *,
        result: dict,
    ):
        """Log the final task result with all metrics."""
        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="task_result",
            experiment=experiment,
            condition=condition,
            task_id=task_id,
            data=result,
        ))

        self._task_data.setdefault(task_id, []).append({
            "event": "task_result", **result
        })

        # Write per-task detail file
        self._flush_task(task_id)

    def log_error(
        self,
        experiment: str,
        condition: str,
        task_id: str,
        error: str,
    ):
        """Log an error."""
        self._write_event(LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="error",
            experiment=experiment,
            condition=condition,
            task_id=task_id,
            data={"error": error},
        ))

    def _flush_task(self, task_id: str):
        """Write accumulated per-task data to a detail file."""
        if task_id in self._task_data:
            safe_id = task_id.replace("/", "_").replace("\\", "_")
            detail_path = self.runs_dir / f"{safe_id}.json"
            with open(detail_path, "w") as f:
                json.dump({
                    "task_id": task_id,
                    "events": self._task_data[task_id],
                }, f, indent=2, default=str)

    def write_summary(self):
        """Write final summary file."""
        elapsed = time.monotonic() - self._start_time
        summary = {
            "run_id": self.run_id,
            "total_events": self._event_count,
            "total_llm_calls": self._llm_call_count,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "elapsed_seconds": round(elapsed, 1),
            "log_dir": str(self.log_dir),
        }
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Log summary: {self._event_count} events, "
              f"{self._llm_call_count} LLM calls, "
              f"{self._total_tokens} tokens, "
              f"${self._total_cost:.4f} cost, "
              f"{elapsed:.1f}s elapsed")
        print(f"  Logs: {self.log_dir}")

    def close(self):
        """Flush and close."""
        self.write_summary()
        self._log_file.close()
