"""Master script to run all experiments sequentially."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    start = time.time()

    print("=" * 70)
    print("FIXING LAZY LLMS - Full Experiment Suite")
    print("=" * 70)

    model = "gpt-4.1-mini"
    n = 200  # problems per experiment

    # Experiment 1: Critique Stringency
    print("\n\n" + "=" * 70)
    print("RUNNING EXPERIMENT 1: Critique Stringency Spectrum")
    print("=" * 70)
    from experiment_critique import run_experiment as run_exp1
    run_exp1(n_problems=n, model=model)

    # Experiment 2: Budget Control
    print("\n\n" + "=" * 70)
    print("RUNNING EXPERIMENT 2: Budget Control")
    print("=" * 70)
    from experiment_budget import run_experiment as run_exp2
    run_exp2(n_problems=n, model=model)

    # Experiment 3: Combined (smaller n since 9 conditions)
    print("\n\n" + "=" * 70)
    print("RUNNING EXPERIMENT 3: Critique x Budget Interaction")
    print("=" * 70)
    from experiment_combined import run_experiment as run_exp3
    run_exp3(n_problems=n, model=model)

    # Experiment 4: Alternative Strategies
    print("\n\n" + "=" * 70)
    print("RUNNING EXPERIMENT 4: Alternative Strategies")
    print("=" * 70)
    from experiment_alternatives import run_experiment as run_exp4
    run_exp4(n_problems=n, model=model)

    # Analysis
    print("\n\n" + "=" * 70)
    print("RUNNING ANALYSIS")
    print("=" * 70)
    from analysis import run_all_analyses
    run_all_analyses(model.replace(".", "_").replace("-", "_"))

    elapsed = time.time() - start
    print(f"\n\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
