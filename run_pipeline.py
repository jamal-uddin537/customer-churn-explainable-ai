"""Run the whole project end to end: data -> clean -> train -> evaluate -> explain.

Usage:
    python run_pipeline.py              # full run (Optuna 40 trials, ~3-4 min)
    python run_pipeline.py --fast       # quick run (10 trials)
    python run_pipeline.py --synthetic  # use synthetic data instead of downloading
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    """Run a project script with the current interpreter and stop on failure."""
    print(f"\n▶ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    """Execute every pipeline stage in order."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast", action="store_true", help="10 Optuna trials instead of 40")
    p.add_argument("--synthetic", action="store_true", help="generate synthetic data")
    a = p.parse_args()
    run("data/get_data.py", *(["--synthetic"] if a.synthetic else []))
    run("src/data_prep.py")
    run("src/train.py", "10" if a.fast else "40")
    run("src/evaluate.py")
    run("src/explain.py")
    print("\n✅ Done. Launch the dashboard with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
