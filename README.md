# LDP Research: Empirical Evaluation of the LLM Delegate Protocol

Experiment code, data, and paper source for:

> **LDP: An Identity-Aware Protocol for Multi-Agent LLM Systems**
> Sunil Prakash, Indian School of Business
> arXiv: [2603.08852](https://arxiv.org/abs/2603.08852v1)

> **The Provenance Paradox in Multi-Agent LLM Routing: Delegation Contracts and Attested Identity in LDP**
> Sunil Prakash, Indian School of Business
> arXiv: [2603.18043](https://arxiv.org/abs/2603.18043)

## Overview

This repository contains everything needed to reproduce the empirical evaluation of the [LLM Delegate Protocol (LDP)](https://github.com/sunilp/ldp-protocol) against A2A and random baselines. The study evaluates six research questions spanning routing quality, payload efficiency, provenance impact, session overhead, security boundaries, and fallback reliability.

## Key Results

| RQ | Finding | Evidence |
|---|---|---|
| RQ1 Routing | ~12x lower latency on easy tasks; quality comparable (n.s.) | Empirical |
| RQ2 Payload | 37% token reduction (p=0.031) with no observed quality loss | Empirical |
| RQ3 Provenance | Noisy provenance degrades quality below no-provenance baseline | Empirical |
| RQ4 Sessions | 39% token overhead eliminated at 10 conversation rounds | Empirical |
| RQ5 Security | 96% vs 6% attack detection rate | Simulated |
| RQ6 Fallback | 100% vs 35% task completion under failures | Simulated |

## Repository Structure

```
ldp-research/
├── paper/                     # Paper 1: LaTeX source, figures, tables
│   ├── main.tex               # Paper source
│   ├── figures/               # 9 publication-quality PDF figures
│   ├── tables/                # 7 LaTeX tables
│   └── generate_figures.py    # Figure generation script
├── paper2/                    # Paper 2: Provenance Paradox (arXiv:2603.18043)
│   ├── main.tex               # Paper source
│   └── figures/               # Publication figures
├── baselines/                 # Protocol baseline implementations
│   ├── protocol.py            # ProtocolBaseline abstract interface
│   ├── ldp_baseline.py        # LDP: metadata routing + identity prompts
│   ├── a2a_baseline.py        # A2A: skill-match routing + generic prompts
│   ├── ablation_baselines.py  # 2x2 factorial ablation conditions
│   └── llm_client.py          # Unified LLM client (Ollama, Gemini)
├── experiments/
│   ├── runners/               # Experiment runner and main entry point
│   ├── evaluation/            # LLM-as-judge, metrics, logging
│   ├── analysis/              # Results analysis and LaTeX table generation
│   └── configs/               # YAML experiment configurations
├── results/
│   ├── tables/                # Aggregated results (JSON)
│   └── logs/                  # Raw experiment logs (JSONL)
├── src/                       # Rust protocol implementation (see ldp-protocol)
├── tests/                     # Integration tests
└── docs/                      # Design documentation
```

## Reproducing Experiments

### Prerequisites

- [Ollama](https://ollama.com/) with models: `qwen3:8b`, `qwen2.5-coder:7b`, `llama3.2:3b`
- Python 3.10+ with dependencies: `pip install -r requirements.txt`
- Google Gemini API key (for LLM-as-judge evaluation)

### Setup

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# Pull Ollama models
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:3b
```

### Run Experiments

```bash
# Run all experiments (local Ollama config)
python -m experiments.runners.main --config experiments/configs/local.yaml

# Run specific experiment
python -m experiments.runners.main --config experiments/configs/local.yaml --experiments routing

# Generate paper figures
python paper/generate_figures.py

# Generate paper tables
python -m experiments.analysis.generate_latex
```

### Hardware

All experiments were run on a single Apple Silicon machine (36GB RAM) using local Ollama inference. Total compute: ~8 hours for all experiments including ablation.

### Related Writing

- [Why Multi-Agent AI Systems Need Identity-Aware Routing](https://sunilprakash.com/writing/ldp-protocol/)
- [From Debate to Deliberation: When Multi-Agent Reasoning Needs Structure](https://sunilprakash.com/writing/deliberative-collective-intelligence/)

## Related Repositories

- **[ldp-protocol](https://github.com/sunilp/ldp-protocol)** — Protocol specification (RFC) and Rust reference implementation
- **[JamJet](https://github.com/jamjet-labs/jamjet)** — Agent runtime that hosts the LDP adapter

## License

Apache-2.0
