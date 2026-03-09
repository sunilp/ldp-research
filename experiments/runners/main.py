"""Main entry point for running LDP research experiments.

Usage:
    # Run all experiments
    python -m experiments.runners.main

    # Run specific experiments
    python -m experiments.runners.main --experiments routing sessions

    # Custom config
    python -m experiments.runners.main --config experiments/configs/experiments.yaml
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if present (for API keys)
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

from experiments.runners.experiment_runner import run_all


def main():
    parser = argparse.ArgumentParser(
        description="Run LDP research experiments"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/experiments.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--jamjet",
        action="store_true",
        help="Run through JamJet runtime (requires `jamjet dev`)",
    )

    args = parser.parse_args()

    print("LDP Research — Experiment Runner")
    print(f"Config: {args.config}")
    print(f"Experiments: {args.experiments or 'all'}")
    if args.jamjet:
        print("Mode: JamJet-integrated (real runtime)")
    else:
        print("Mode: Simulated baselines")
    print()

    asyncio.run(run_all(args.config, args.experiments))


if __name__ == "__main__":
    main()
