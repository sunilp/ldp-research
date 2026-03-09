"""Generate publication-quality figures for LDP research paper.

Run: python paper/generate_figures.py
Output: paper/figures/*.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "ldp": "#2563EB",       # Blue
    "a2a": "#DC2626",       # Red
    "random": "#9CA3AF",    # Gray
    "text": "#6366F1",      # Indigo
    "semantic": "#2563EB",  # Blue
    "a2a_json": "#DC2626",  # Red
    "with": "#2563EB",      # Blue
    "without": "#9CA3AF",   # Gray
    "noisy": "#DC2626",     # Red
}


def load_table(name):
    p = RESULTS / "tables" / f"{name}.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_events(experiment):
    logs = RESULTS / "logs"
    if not logs.exists():
        return []
    for d in sorted(logs.iterdir(), reverse=True):
        ef = d / "events.jsonl"
        if ef.exists():
            with open(ef) as f:
                first = f.readline()
                if first:
                    e = json.loads(first)
                    if e.get("experiment") == experiment:
                        f.seek(0)
                        return [json.loads(l) for l in f]
    return []


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Protocol Architecture Diagram
# ══════════════════════════════════════════════════════════════════════

def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Router box
    router = mpatches.FancyBboxPatch((3.5, 4.2), 3, 1.2, boxstyle="round,pad=0.15",
                                      facecolor="#DBEAFE", edgecolor="#2563EB", linewidth=1.5)
    ax.add_patch(router)
    ax.text(5, 4.8, "LDP Router", ha="center", va="center", fontweight="bold", fontsize=11, color="#1E40AF")

    # Identity cards
    labels = [
        ("qwen3:8b\nq=0.85, deep-analytical", 0.3, 1.8, "#EFF6FF"),
        ("qwen2.5-coder:7b\nq=0.80, code-specialist", 3.5, 1.8, "#EFF6FF"),
        ("llama3.2:3b\nq=0.55, fast-practical", 6.7, 1.8, "#EFF6FF"),
    ]
    for txt, x, y, color in labels:
        box = mpatches.FancyBboxPatch((x, y), 2.8, 1.4, boxstyle="round,pad=0.12",
                                       facecolor=color, edgecolor="#93C5FD", linewidth=1)
        ax.add_patch(box)
        ax.text(x + 1.4, y + 0.7, txt, ha="center", va="center", fontsize=7.5, color="#1E3A5F")

    # Arrows from router to delegates
    arrow_style = dict(arrowstyle="->,head_width=0.15", color="#6B7280", lw=1.2)
    ax.annotate("", xy=(1.7, 3.2), xytext=(4.2, 4.2), arrowprops=arrow_style)
    ax.annotate("", xy=(4.9, 3.2), xytext=(5, 4.2), arrowprops=arrow_style)
    ax.annotate("", xy=(8.1, 3.2), xytext=(5.8, 4.2), arrowprops=arrow_style)

    # Labels on arrows
    ax.text(2.5, 3.9, "hard tasks", fontsize=7, color="#4B5563", rotation=25, ha="center")
    ax.text(5.15, 3.7, "code", fontsize=7, color="#4B5563", ha="center")
    ax.text(7.4, 3.9, "easy tasks", fontsize=7, color="#4B5563", rotation=-25, ha="center")

    # Protocol features below
    features = [
        ("Identity\nCards", 0.8),
        ("Payload\nNegotiation", 2.8),
        ("Governed\nSessions", 4.8),
        ("Provenance\nTracking", 6.8),
        ("Trust\nDomains", 8.8),
    ]
    for txt, x in features:
        box = mpatches.FancyBboxPatch((x - 0.7, 0.1), 1.4, 0.9, boxstyle="round,pad=0.08",
                                       facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=0.8)
        ax.add_patch(box)
        ax.text(x, 0.55, txt, ha="center", va="center", fontsize=7, color="#166534")

    # Title
    ax.text(5, 5.7, "LDP Protocol Architecture", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#111827")

    fig.savefig(FIGURES / "fig_architecture.pdf")
    plt.close(fig)
    print("  Generated: fig_architecture.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Routing Quality by Difficulty (grouped bar chart)
# ══════════════════════════════════════════════════════════════════════

def fig_routing_quality():
    events = load_events("routing")
    if not events:
        print("  SKIP: fig_routing_quality (no data)")
        return

    results = [e for e in events if e.get("event_type") == "task_result"]

    # Collect data
    data = {}
    for cond in ["ldp", "a2a", "random"]:
        data[cond] = {}
        for r in results:
            if r.get("condition") == cond:
                d = r["data"].get("difficulty", "?")
                data[cond].setdefault(d, [])
                data[cond][d].append(r["data"]["quality_score"])

    difficulties = ["easy", "medium", "hard"]
    conditions = ["ldp", "a2a", "random"]
    cond_labels = {"ldp": "LDP", "a2a": "A2A", "random": "Random"}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x = np.arange(len(difficulties))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(conditions):
        means = []
        stds = []
        for d in difficulties:
            vals = data[cond].get(d, [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(x + offsets[i], means, width, yerr=stds,
                      label=cond_labels[cond], color=COLORS[cond],
                      capsize=3, edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 0.8}, alpha=0.9)

        # Value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Task Difficulty")
    ax.set_ylabel("Quality Score (1-10)")
    ax.set_title("Routing Quality by Task Difficulty (RQ1)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, 12)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.savefig(FIGURES / "fig_routing_quality.pdf")
    plt.close(fig)
    print("  Generated: fig_routing_quality.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Routing Latency by Difficulty
# ══════════════════════════════════════════════════════════════════════

def fig_routing_latency():
    events = load_events("routing")
    if not events:
        print("  SKIP: fig_routing_latency (no data)")
        return

    results = [e for e in events if e.get("event_type") == "task_result"]

    data = {}
    for cond in ["ldp", "a2a", "random"]:
        data[cond] = {}
        for r in results:
            if r.get("condition") == cond:
                d = r["data"].get("difficulty", "?")
                data[cond].setdefault(d, [])
                data[cond][d].append(r["data"]["latency_ms"] / 1000)

    difficulties = ["easy", "medium", "hard"]
    conditions = ["ldp", "a2a", "random"]
    cond_labels = {"ldp": "LDP", "a2a": "A2A", "random": "Random"}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x = np.arange(len(difficulties))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(conditions):
        means = []
        stds = []
        for d in difficulties:
            vals = data[cond].get(d, [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(x + offsets[i], means, width, yerr=stds,
                      label=cond_labels[cond], color=COLORS[cond],
                      capsize=3, edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 0.8}, alpha=0.9)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{m:.1f}s", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Task Difficulty")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Routing Latency by Task Difficulty (RQ1)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(loc="upper left", framealpha=0.9)

    fig.savefig(FIGURES / "fig_routing_latency.pdf")
    plt.close(fig)
    print("  Generated: fig_routing_latency.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: Payload Efficiency (tokens + quality side by side)
# ══════════════════════════════════════════════════════════════════════

def fig_payload():
    data = load_table("payload")
    if not data:
        print("  SKIP: fig_payload (no data)")
        return

    labels_map = {
        "mode0_text": "Text\n(Mode 0)",
        "mode1_semantic_frame": "Semantic Frame\n(Mode 1)",
        "a2a_json": "A2A JSON",
    }
    color_map = {
        "mode0_text": COLORS["text"],
        "mode1_semantic_frame": COLORS["semantic"],
        "a2a_json": COLORS["a2a_json"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    labels = [labels_map.get(c["condition"], c["condition"]) for c in data]
    colors = [color_map.get(c["condition"], "#999") for c in data]

    # Token count
    tok_means = [c["metrics"]["token_count"]["mean"] for c in data]
    tok_stds = [c["metrics"]["token_count"]["std"] for c in data]
    bars1 = ax1.bar(labels, tok_means, yerr=tok_stds, color=colors,
                    capsize=4, edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 0.8}, alpha=0.9)
    for bar, m in zip(bars1, tok_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f"{m:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Token Count")
    ax1.set_title("Token Efficiency")

    # Add reduction annotation
    ax1.annotate("−37%", xy=(1, tok_means[1]), xytext=(1.4, tok_means[0] - 100),
                 fontsize=9, fontweight="bold", color="#059669",
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5))

    # Quality
    q_means = [c["metrics"]["output_quality"]["mean"] for c in data]
    q_stds = [c["metrics"]["output_quality"]["std"] for c in data]
    bars2 = ax2.bar(labels, q_means, yerr=q_stds, color=colors,
                    capsize=4, edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 0.8}, alpha=0.9)
    for bar, m in zip(bars2, q_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Output Quality (1-10)")
    ax2.set_title("Quality Comparison")
    ax2.set_ylim(0, 10)

    fig.suptitle("Payload Mode Efficiency (RQ2)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_payload.pdf")
    plt.close(fig)
    print("  Generated: fig_payload.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 5: Provenance Impact
# ══════════════════════════════════════════════════════════════════════

def fig_provenance():
    data = load_table("provenance")
    if not data:
        print("  SKIP: fig_provenance (no data)")
        return

    labels_map = {
        "with_provenance": "Accurate\nProvenance",
        "without_provenance": "No\nProvenance",
        "noisy_provenance": "Noisy\nProvenance",
    }
    color_map = {
        "with_provenance": COLORS["with"],
        "without_provenance": COLORS["without"],
        "noisy_provenance": COLORS["noisy"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))

    labels = [labels_map.get(c["condition"], c["condition"]) for c in data]
    colors = [color_map.get(c["condition"], "#999") for c in data]

    # Decision quality
    dq_means = [c["metrics"]["decision_quality"]["mean"] for c in data]
    dq_stds = [c["metrics"]["decision_quality"]["std"] for c in data]
    bars1 = ax1.bar(labels, dq_means, yerr=dq_stds, color=colors,
                    capsize=4, edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 0.8}, alpha=0.9)
    for bar, m in zip(bars1, dq_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Decision Quality (1-10)")
    ax1.set_title("Decision Quality")
    ax1.set_ylim(0, 11)

    # Calibration accuracy
    ca_means = [c["metrics"]["calibration_accuracy"]["mean"] for c in data]
    ca_stds = [c["metrics"]["calibration_accuracy"]["std"] for c in data]
    bars2 = ax2.bar(labels, ca_means, yerr=ca_stds, color=colors,
                    capsize=4, edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 0.8}, alpha=0.9)
    for bar, m in zip(bars2, ca_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Calibration Accuracy (1-10)")
    ax2.set_title("Source Calibration")
    ax2.set_ylim(0, 12)

    fig.suptitle("Provenance Impact on Synthesis Quality (RQ3)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_provenance.pdf")
    plt.close(fig)
    print("  Generated: fig_provenance.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 6: Security + Fallback (combined panel)
# ══════════════════════════════════════════════════════════════════════

def fig_security_fallback():
    sec = load_table("security")
    fb = load_table("fallback")
    if not sec or not fb:
        print("  SKIP: fig_security_fallback (no data)")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2))

    # Security: Detection Rate
    sec_labels = ["LDP\nTrust Domains", "A2A\nBearer Token"]
    sec_detect = [sec[0]["metrics"]["detection_rate"]["mean"] * 100,
                  sec[1]["metrics"]["detection_rate"]["mean"] * 100]
    sec_colors = [COLORS["ldp"], COLORS["a2a"]]
    bars1 = ax1.bar(sec_labels, sec_detect, color=sec_colors,
                    edgecolor="white", linewidth=0.5, alpha=0.9)
    for bar, m in zip(bars1, sec_detect):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{m:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Detection Rate (%)")
    ax1.set_title("Attack Detection (RQ5)")
    ax1.set_ylim(0, 115)

    # Fallback: Completion rate
    fb_labels = ["LDP\n(fallback)", "A2A\n(no fallback)"]
    fb_comp = [fb[0]["metrics"]["completion_rate"]["mean"] * 100,
               fb[1]["metrics"]["completion_rate"]["mean"] * 100]
    fb_colors = [COLORS["ldp"], COLORS["a2a"]]
    bars2 = ax2.bar(fb_labels, fb_comp, color=fb_colors,
                    edgecolor="white", linewidth=0.5, alpha=0.9)
    for bar, m in zip(bars2, fb_comp):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{m:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Completion Rate (%)")
    ax2.set_title("Task Completion (RQ6)")
    ax2.set_ylim(0, 115)

    # Fallback: Quality degradation
    fb_qd_means = [fb[0]["metrics"]["quality_degradation"]["mean"],
                   fb[1]["metrics"]["quality_degradation"]["mean"]]
    fb_qd_stds = [fb[0]["metrics"]["quality_degradation"]["std"],
                  fb[1]["metrics"]["quality_degradation"]["std"]]
    bars3 = ax3.bar(fb_labels, fb_qd_means, yerr=fb_qd_stds, color=fb_colors,
                    capsize=4, edgecolor="white", linewidth=0.5,
                    error_kw={"linewidth": 0.8}, alpha=0.9)
    for bar, m in zip(bars3, fb_qd_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Quality Degradation (0-1)")
    ax3.set_title("Quality Loss (RQ6)")
    ax3.set_ylim(0, 1.3)

    fig.suptitle("Security and Fallback Evaluation (RQ5, RQ6)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_security_fallback.pdf")
    plt.close(fig)
    print("  Generated: fig_security_fallback.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 7: Summary Comparison (horizontal bar chart)
# ══════════════════════════════════════════════════════════════════════

def fig_summary():
    metrics = [
        "Easy Task\nLatency (×)",
        "Token\nReduction (%)",
        "Session Overhead\nSaved (%)",
        "Overall\nQuality",
        "Attack\nDetection (%)",
        "Completion\nRate (%)",
    ]
    ldp_vals = [12.0, 37, 39, 6.8, 96, 100]
    a2a_vals = [1.0, 7, 0, 7.4, 6, 35]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    y = np.arange(len(metrics))
    height = 0.35

    bars1 = ax.barh(y - height / 2, ldp_vals, height, label="LDP",
                    color=COLORS["ldp"], edgecolor="white", linewidth=0.5, alpha=0.9)
    bars2 = ax.barh(y + height / 2, a2a_vals, height, label="A2A",
                    color=COLORS["a2a"], edgecolor="white", linewidth=0.5, alpha=0.9)

    # Value labels
    for bar, v in zip(bars1, ldp_vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v}", va="center", fontsize=8, fontweight="bold", color=COLORS["ldp"])
    for bar, v in zip(bars2, a2a_vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v}", va="center", fontsize=8, fontweight="bold", color=COLORS["a2a"])

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Score / Percentage")
    ax.set_title("LDP vs A2A: Summary Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 115)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_summary.pdf")
    plt.close(fig)
    print("  Generated: fig_summary.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 8: Protocol Comparison Table (visual)
# ══════════════════════════════════════════════════════════════════════

def fig_protocol_comparison():
    features = [
        "Model Identity",
        "Quality Hints",
        "Cost Profiles",
        "Reasoning Profile",
        "Payload Negotiation",
        "Governed Sessions",
        "Provenance Tracking",
        "Trust Domains",
        "Fallback Chain",
        "Multi-Party Rooms",
    ]

    # 1 = supported, 0.5 = partial, 0 = not supported
    ldp_support = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    a2a_support = [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0]
    mcp_support = [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    y = np.arange(len(features))
    height = 0.25

    ax.barh(y - height, ldp_support, height, label="LDP", color=COLORS["ldp"], alpha=0.9, edgecolor="white")
    ax.barh(y, a2a_support, height, label="A2A", color=COLORS["a2a"], alpha=0.9, edgecolor="white")
    ax.barh(y + height, mcp_support, height, label="MCP", color="#F59E0B", alpha=0.9, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["Not\nSupported", "Partial", "Full\nSupport"])
    ax.set_title("Protocol Feature Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_protocol_comparison.pdf")
    plt.close(fig)
    print("  Generated: fig_protocol_comparison.pdf")


def fig_sessions():
    """Figure: Session efficiency — token overhead scaling."""
    rounds = [3, 5, 10]
    ldp_tokens = [3860, 6379, 12990]
    a2a_tokens = [3847, 6798, 16010]
    a2a_overhead_pct = [10.4, 20.2, 39.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Left: Total tokens
    x = np.arange(len(rounds))
    w = 0.35
    bars1 = ax1.bar(x - w/2, ldp_tokens, w, label="LDP (sessions)",
                    color=COLORS["ldp"], edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, a2a_tokens, w, label="A2A (stateless)",
                    color=COLORS["a2a"], edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Conversation Rounds")
    ax1.set_ylabel("Total Tokens")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in rounds])
    ax1.legend(framealpha=0.9)
    ax1.set_title("Token Count by Rounds")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 200,
                 f'{int(h):,}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 200,
                 f'{int(h):,}', ha='center', va='bottom', fontsize=7)

    # Right: Overhead percentage
    ax2.plot(rounds, [1.6, 0, 0], 'o-', color=COLORS["ldp"],
             label="LDP overhead", linewidth=2, markersize=6)
    ax2.plot(rounds, a2a_overhead_pct, 's-', color=COLORS["a2a"],
             label="A2A overhead", linewidth=2, markersize=6)
    ax2.fill_between(rounds, a2a_overhead_pct, alpha=0.15, color=COLORS["a2a"])
    ax2.set_xlabel("Conversation Rounds")
    ax2.set_ylabel("Overhead Tokens (%)")
    ax2.set_xticks(rounds)
    ax2.legend(framealpha=0.9)
    ax2.set_title("Context Re-transmission Overhead")
    ax2.set_ylim(-2, 50)

    # Annotate the 39% point
    ax2.annotate('39%', xy=(10, 39.2), xytext=(8.5, 45),
                 fontsize=9, fontweight='bold', color=COLORS["a2a"],
                 arrowprops=dict(arrowstyle='->', color=COLORS["a2a"]))

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_sessions.pdf")
    plt.close(fig)
    print("  Generated: fig_sessions.pdf")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")
    fig_architecture()
    fig_routing_quality()
    fig_routing_latency()
    fig_payload()
    fig_provenance()
    fig_security_fallback()
    fig_sessions()
    fig_summary()
    fig_protocol_comparison()
    print(f"\nAll figures saved to: {FIGURES}/")
