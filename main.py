"""
main.py — one-command runner for the WP0 falsification.

Run this from the repo root (the folder containing src/, analysis/, tests/):

    python main.py

It will:
  1. Run the falsification experiment (analysis/falsify.py) and write
     results/results.json + results/run.npz
  2. Generate the summary figure (analysis/make_figure.py) and write
     results/fig0_no_rolling.png
  3. Print a short summary so you can see the key numbers immediately.

If you just want to check everything still works after editing code,
run the test suite instead:

    pytest tests/ -q
"""

import os
import sys
import json
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
ANALYSIS = os.path.join(ROOT, "analysis")
RESULTS = os.path.join(ROOT, "results")

# Make sure src/ and analysis/ are importable no matter where this is run from.
for p in (SRC, ANALYSIS):
    if p not in sys.path:
        sys.path.insert(0, p)


def run_step(label, script_path):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    runpy.run_path(script_path, run_name="__main__")


def print_summary():
    results_file = os.path.join(RESULTS, "results.json")
    if not os.path.exists(results_file):
        print("\n(no results.json found to summarize)")
        return
    with open(results_file, "r") as f:
        R = json.load(f)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for key, val in R.items():
        print(f"  {key}: {val}")


def main():
    falsify_script = os.path.join(ANALYSIS, "falsify.py")
    figure_script = os.path.join(ANALYSIS, "make_figure.py")

    if not os.path.exists(falsify_script):
        sys.exit(f"ERROR: could not find {falsify_script}. "
                  f"Run this from the repo root.")

    run_step("Running falsification experiment (falsify.py)", falsify_script)

    if os.path.exists(figure_script):
        run_step("Generating summary figure (make_figure.py)", figure_script)
    else:
        print(f"\n(skipping figure — {figure_script} not found)")

    print_summary()

    fig_path = os.path.join(RESULTS, "fig0_no_rolling.png")
    print(f"\nDone. Check results/results.json and, if generated, "
          f"{os.path.relpath(fig_path, ROOT)}\n")


if __name__ == "__main__":
    main()
