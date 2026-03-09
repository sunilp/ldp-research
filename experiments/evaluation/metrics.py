"""Metrics computation for experiment results.

Aggregates per-run metrics into summary statistics with
confidence intervals for the paper's tables and figures.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricSummary:
    """Summary statistics for a single metric across multiple runs."""
    name: str
    values: list[float]
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    ci_95_low: float = 0.0
    ci_95_high: float = 0.0
    min: float = 0.0
    max: float = 0.0
    n: int = 0

    def compute(self) -> MetricSummary:
        """Compute summary statistics from values."""
        if not self.values:
            return self

        self.n = len(self.values)
        self.mean = statistics.mean(self.values)
        self.median = statistics.median(self.values)
        self.min = min(self.values)
        self.max = max(self.values)

        if self.n > 1:
            self.std = statistics.stdev(self.values)
            # 95% CI using t-distribution approximation
            se = self.std / math.sqrt(self.n)
            t_val = 1.96 if self.n >= 30 else 2.0  # Rough approximation
            self.ci_95_low = self.mean - t_val * se
            self.ci_95_high = self.mean + t_val * se
        else:
            self.std = 0.0
            self.ci_95_low = self.mean
            self.ci_95_high = self.mean

        return self

    @property
    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "ci_95": [round(self.ci_95_low, 4), round(self.ci_95_high, 4)],
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "n": self.n,
        }


@dataclass
class ExperimentResults:
    """Aggregated results for one experiment condition."""
    experiment: str
    condition: str
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    raw_runs: list[dict[str, Any]] = field(default_factory=list)

    def add_run(self, run: dict[str, Any]):
        """Add a single run's data."""
        self.raw_runs.append(run)
        for key, value in run.items():
            if isinstance(value, (int, float)):
                if key not in self.metrics:
                    self.metrics[key] = MetricSummary(name=key, values=[])
                self.metrics[key].values.append(float(value))

    def compute_all(self):
        """Compute summary statistics for all metrics."""
        for metric in self.metrics.values():
            metric.compute()

    @property
    def as_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "condition": self.condition,
            "n_runs": len(self.raw_runs),
            "metrics": {k: v.as_dict for k, v in self.metrics.items()},
        }


def compute_effect_size(treatment: MetricSummary, control: MetricSummary) -> float:
    """Compute Cohen's d effect size between treatment and control."""
    if treatment.n < 2 or control.n < 2:
        return 0.0

    pooled_std = math.sqrt(
        ((treatment.n - 1) * treatment.std**2 + (control.n - 1) * control.std**2)
        / (treatment.n + control.n - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (treatment.mean - control.mean) / pooled_std


def significance_test(
    treatment_values: list[float], control_values: list[float]
) -> dict:
    """Mann-Whitney U test for significance (non-parametric).

    Returns dict with u_statistic, p_value, significant (at 0.05).
    Falls back to simple comparison if scipy unavailable.
    """
    try:
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(
            treatment_values, control_values, alternative="two-sided"
        )
        return {
            "test": "mann_whitney_u",
            "u_statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    except ImportError:
        # Fallback: simple mean comparison
        t_mean = statistics.mean(treatment_values) if treatment_values else 0
        c_mean = statistics.mean(control_values) if control_values else 0
        return {
            "test": "mean_comparison",
            "treatment_mean": t_mean,
            "control_mean": c_mean,
            "difference": t_mean - c_mean,
            "significant": None,  # Can't determine without scipy
        }


def format_results_table(
    results: list[ExperimentResults], metric_names: list[str]
) -> str:
    """Format results as a markdown table for the paper."""
    if not results or not metric_names:
        return ""

    header = "| Condition | " + " | ".join(metric_names) + " |"
    separator = "|---|" + "|".join(["---"] * len(metric_names)) + "|"

    rows = []
    for r in results:
        cells = [r.condition]
        for name in metric_names:
            if name in r.metrics:
                m = r.metrics[name]
                cells.append(f"{m.mean:.2f} ± {m.std:.2f}")
            else:
                cells.append("—")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)
