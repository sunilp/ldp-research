"""Generate LaTeX tables from experiment results.

Run: python -m experiments.analysis.generate_latex
Output: paper/tables/*.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

TABLES_DIR = project_root / "paper" / "tables"


def load_table(name: str) -> list[dict]:
    path = project_root / "results" / "tables" / f"{name}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_log_events(experiment: str) -> list[dict]:
    """Find and load log events for an experiment."""
    logs_dir = project_root / "results" / "logs"
    if not logs_dir.exists():
        return []
    for d in sorted(logs_dir.iterdir(), reverse=True):
        events_file = d / "events.jsonl"
        if events_file.exists():
            with open(events_file) as f:
                first_line = f.readline()
                if first_line:
                    e = json.loads(first_line)
                    if e.get("experiment") == experiment:
                        f.seek(0)
                        return [json.loads(line) for line in f]
    return []


def fmt(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


def fmt_pm(mean: float, std: float, decimals: int = 2) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def generate_routing_tables():
    """Table 1: Routing quality overview + Table 2: By difficulty."""
    data = load_table("routing")
    if not data:
        return

    # Table 1: Overview
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Routing quality comparison across protocol conditions (RQ1). "
        r"Quality scores are LLM-judge ratings (1--10). All delegates run locally via Ollama.}",
        r"\label{tab:routing-overview}",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"Condition & Quality & Latency (s) & Tokens & $n$ \\",
        r"\midrule",
    ]

    for c in data:
        name = c["condition"].upper()
        q = c["metrics"].get("quality_score", {})
        l = c["metrics"].get("latency_ms", {})
        t = c["metrics"].get("total_tokens", {})
        n = q.get("n", 0)
        lines.append(
            f"{name} & {fmt_pm(q.get('mean',0), q.get('std',0))} & "
            f"{fmt(l.get('mean',0)/1000, 1)} & "
            f"{fmt(t.get('mean',0), 0)} & {n} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "tab_routing_overview.tex").write_text("\n".join(lines))

    # Table 2: By difficulty
    events = load_log_events("routing")
    if not events:
        return

    results = [e for e in events if e.get("event_type") == "task_result"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Routing quality by task difficulty (RQ1). LDP routes easy tasks "
        r"to a lightweight delegate (llama-3b, 1s latency) while maintaining quality. "
        r"A2A always selects the strongest model but scores lowest on hard tasks due to "
        r"its generic system prompt.}",
        r"\label{tab:routing-difficulty}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Easy} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Hard} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Condition & Quality & Latency & Quality & Latency & Quality & Latency \\",
        r"\midrule",
    ]

    for cond in ["ldp", "a2a", "random"]:
        by_diff = {}
        for r in results:
            if r.get("condition") == cond:
                d = r["data"].get("difficulty", "?")
                by_diff.setdefault(d, {"q": [], "l": []})
                by_diff[d]["q"].append(r["data"]["quality_score"])
                by_diff[d]["l"].append(r["data"]["latency_ms"])

        cells = [cond.upper()]
        for diff in ["easy", "medium", "hard"]:
            if diff in by_diff:
                q_avg = sum(by_diff[diff]["q"]) / len(by_diff[diff]["q"])
                l_avg = sum(by_diff[diff]["l"]) / len(by_diff[diff]["l"]) / 1000
                cells.append(fmt(q_avg, 1))
                cells.append(f"{fmt(l_avg, 1)}s")
            else:
                cells += ["--", "--"]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "tab_routing_difficulty.tex").write_text("\n".join(lines))
    print("  Generated: tab_routing_overview.tex, tab_routing_difficulty.tex")


def generate_payload_table():
    """Table 3: Payload mode comparison."""
    data = load_table("payload")
    if not data:
        return

    mode_labels = {
        "mode0_text": "Text (Mode 0)",
        "mode1_semantic_frame": "Semantic Frame (Mode 1)",
        "a2a_json": "A2A JSON",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Payload mode efficiency comparison (RQ2). Semantic frames "
        r"reduce token count by 37\% and latency by 42\% compared to raw text, "
        r"with no observed quality loss ($p=0.96$). Token reduction is statistically significant ($p=0.031$, $d=-0.7$).}",
        r"\label{tab:payload}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Mode & Tokens & Latency (s) & Quality & Info Preserved \\",
        r"\midrule",
    ]

    text_tok = None
    for c in data:
        name = mode_labels.get(c["condition"], c["condition"])
        tok = c["metrics"].get("token_count", {})
        lat = c["metrics"].get("wall_clock_ms", {})
        qual = c["metrics"].get("output_quality", {})
        info = c["metrics"].get("information_preserved", {})

        tok_mean = tok.get("mean", 0)
        if c["condition"] == "mode0_text":
            text_tok = tok_mean

        lines.append(
            f"{name} & {fmt_pm(tok_mean, tok.get('std',0), 0)} & "
            f"{fmt(lat.get('mean',0)/1000, 1)} & "
            f"{fmt_pm(qual.get('mean',0), qual.get('std',0))} & "
            f"{fmt_pm(info.get('mean',0), info.get('std',0))} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "tab_payload.tex").write_text("\n".join(lines))
    print("  Generated: tab_payload.tex")


def generate_provenance_table():
    """Table 4: Provenance impact."""
    data = load_table("provenance")
    if not data:
        return

    cond_labels = {
        "with_provenance": "With Provenance",
        "without_provenance": "Without Provenance",
        "noisy_provenance": "Noisy Provenance",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Provenance impact on synthesis quality (RQ3). Accurate provenance "
        r"provides modest quality improvement, but noisy (misleading) provenance "
        r"degrades quality and increases variance, arguing for LDP's structured "
        r"verification fields.}",
        r"\label{tab:provenance}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & Decision Quality & Calibration & Synth. Tokens \\",
        r"\midrule",
    ]

    for c in data:
        name = cond_labels.get(c["condition"], c["condition"])
        dq = c["metrics"].get("decision_quality", {})
        ca = c["metrics"].get("calibration_accuracy", {})
        st = c["metrics"].get("synthesis_tokens", {})

        lines.append(
            f"{name} & {fmt_pm(dq.get('mean',0), dq.get('std',0))} & "
            f"{fmt_pm(ca.get('mean',0), ca.get('std',0))} & "
            f"{fmt(st.get('mean',0), 0)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "tab_provenance.tex").write_text("\n".join(lines))
    print("  Generated: tab_provenance.tex")


def generate_security_table():
    """Table 5: Security boundary enforcement."""
    data = load_table("security")
    if not data:
        return

    cond_labels = {
        "ldp_trust_domains": "LDP Trust Domains",
        "a2a_auth": "A2A Bearer Token",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Security boundary enforcement (RQ5, simulated). LDP's trust "
        r"domains detect 96\% of unauthorized delegation attempts with 0\% false "
        r"positives. A2A's bearer-token auth detects only 6\%.}",
        r"\label{tab:security}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Condition & Detection Rate & False Positive Rate \\",
        r"\midrule",
    ]

    for c in data:
        name = cond_labels.get(c["condition"], c["condition"])
        dr = c["metrics"].get("detection_rate", {})
        fpr = c["metrics"].get("false_positive_rate", {})

        lines.append(
            f"{name} & {fmt(dr.get('mean',0)*100, 1)}\\% & "
            f"{fmt(fpr.get('mean',0)*100, 1)}\\% \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "tab_security.tex").write_text("\n".join(lines))
    print("  Generated: tab_security.tex")


def generate_fallback_table():
    """Table 6: Fallback reliability."""
    data = load_table("fallback")
    if not data:
        return

    cond_labels = {
        "ldp_fallback": "LDP (with fallback)",
        "a2a_no_fallback": "A2A (no fallback)",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fallback reliability under communication failures (RQ6). LDP's "
        r"mode fallback chain achieves 100\% task completion with minor quality "
        r"degradation (0.16). A2A has no fallback---failures are terminal.}",
        r"\label{tab:fallback}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & Completion Rate & Quality Degradation & Recovery (ms) \\",
        r"\midrule",
    ]

    for c in data:
        name = cond_labels.get(c["condition"], c["condition"])
        cr = c["metrics"].get("completion_rate", {})
        qd = c["metrics"].get("quality_degradation", {})
        rl = c["metrics"].get("recovery_latency_ms", {})

        lines.append(
            f"{name} & {fmt(cr.get('mean',0)*100, 0)}\\% & "
            f"{fmt_pm(qd.get('mean',0), qd.get('std',0))} & "
            f"{fmt(rl.get('mean',0), 0)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (TABLES_DIR / "tab_fallback.tex").write_text("\n".join(lines))
    print("  Generated: tab_fallback.tex")


def generate_summary_table():
    """Summary table: LDP vs A2A across all dimensions."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Summary of LDP vs.\ A2A across all research questions. "
        r"$\uparrow$ indicates higher is better, $\downarrow$ lower is better. "
        r"Bold indicates the better result. "
        r"$^*$Statistically significant ($p<0.05$). "
        r"$^\dagger$Simulated (protocol design property). "
        r"$^\ddagger$Not statistically significant.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{llcccl}",
        r"\toprule",
        r"RQ & Metric & LDP & A2A & $\Delta$ & Evidence \\",
        r"\midrule",
    ]

    rows = [
        ("RQ1", r"Overall quality$^\ddagger$ $\uparrow$", "6.80", r"\textbf{7.43}", r"$-0.63$", "Empirical"),
        ("", r"Easy task latency $\downarrow$", r"\textbf{2.9s}", "34.8s", r"$12\times$", "Empirical"),
        ("", r"Hard task quality$^\ddagger$ $\uparrow$", "3.3", r"\textbf{4.1}", r"$-0.8$", "Empirical"),
        (r"\midrule", "", "", "", "", ""),
        ("RQ2", r"Token count$^*$ $\downarrow$", r"\textbf{765}", "1128", r"$-32\%$", "Empirical"),
        ("", r"Latency $\downarrow$", r"\textbf{14.0s}", "23.4s", r"$-40\%$", "Empirical"),
        ("", r"Quality$^\ddagger$ $\uparrow$", r"\textbf{5.70}", "5.47", "+0.23", "Empirical"),
        (r"\midrule", "", "", "", "", ""),
        ("RQ3", r"Decision quality$^\ddagger$ $\uparrow$", "7.65", r"\textbf{7.85}", r"$-0.20$", "Empirical"),
        ("", r"Noisy provenance risk $\downarrow$", r"N/A", r"N/A", r"$-1.0$ quality", "Empirical"),
        (r"\midrule", "", "", "", "", ""),
        ("RQ4", r"Token overhead (10 rounds) $\downarrow$", r"\textbf{0\%}", "39\%", r"$-39\text{pp}$", "Empirical"),
        ("", r"Total tokens (10 rounds) $\downarrow$", r"\textbf{12,990}", "16,010", r"$-23\%$", "Empirical"),
        (r"\midrule", "", "", "", "", ""),
        ("RQ5", r"Attack detection$^\dagger$ $\uparrow$", r"\textbf{96\%}", r"6\%", r"$+90\text{pp}$", "Simulated"),
        (r"\midrule", "", "", "", "", ""),
        ("RQ6", r"Completion rate$^\dagger$ $\uparrow$", r"\textbf{100\%}", r"35\%", r"$+65\text{pp}$", "Simulated"),
        ("", r"Quality degradation$^\dagger$ $\downarrow$", r"\textbf{0.16}", "0.67", r"$-0.51$", "Simulated"),
    ]

    for row in rows:
        if row[0] == r"\midrule":
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    (TABLES_DIR / "tab_summary.tex").write_text("\n".join(lines))
    print("  Generated: tab_summary.tex")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating LaTeX tables...")
    generate_routing_tables()
    generate_payload_table()
    generate_provenance_table()
    generate_security_table()
    generate_fallback_table()
    generate_summary_table()

    print(f"\nAll tables written to: {TABLES_DIR}/")
    print("Include in paper with: \\input{tables/tab_routing_overview.tex}")


if __name__ == "__main__":
    main()
