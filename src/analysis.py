"""Analysis and visualization for all experiments."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_results(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"Warning: {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def compute_accuracy_ci(results: list, confidence: float = 0.95) -> tuple:
    """Compute accuracy with Wilson score confidence interval."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return 0, 0, 0, 0
    n = len(valid)
    k = sum(1 for r in valid if r.get("correct", False))
    p = k / n

    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return p, center - margin, center + margin, n


def mcnemar_test(results_a: list, results_b: list) -> dict:
    """McNemar's test for paired binary outcomes."""
    valid_a = [r for r in results_a if "error" not in r]
    valid_b = [r for r in results_b if "error" not in r]
    n = min(len(valid_a), len(valid_b))

    # Build contingency: a_correct/b_correct, a_correct/b_wrong, etc.
    b_val = 0  # a correct, b wrong
    c_val = 0  # a wrong, b correct
    for i in range(n):
        a_correct = valid_a[i].get("correct", False)
        b_correct = valid_b[i].get("correct", False)
        if a_correct and not b_correct:
            b_val += 1
        elif not a_correct and b_correct:
            c_val += 1

    # McNemar's test (with continuity correction)
    if b_val + c_val == 0:
        return {"statistic": 0, "p_value": 1.0, "b": b_val, "c": c_val}

    statistic = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return {"statistic": statistic, "p_value": p_value, "b": b_val, "c": c_val}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def analyze_experiment1(model_suffix: str = "gpt-4_1-mini"):
    """Analyze and visualize Experiment 1: Critique Stringency."""
    data = load_results(f"experiment1_critique_{model_suffix}.json")
    if not data:
        return None

    print("=" * 60)
    print("EXPERIMENT 1: Critique Stringency Spectrum")
    print("=" * 60)

    levels = list(data.keys())
    accuracies = []
    ci_low = []
    ci_high = []
    ns = []
    change_rates = []
    c2w_rates = []
    w2c_rates = []
    avg_tokens = []

    for level in levels:
        results = data[level]
        acc, lo, hi, n = compute_accuracy_ci(results)
        accuracies.append(acc)
        ci_low.append(lo)
        ci_high.append(hi)
        ns.append(n)

        valid = [r for r in results if "error" not in r]
        change_rates.append(
            sum(1 for r in valid if r.get("answer_changed", False)) / max(1, len(valid))
        )

        initially_correct = [r for r in valid if r.get("initial_correct", False)]
        initially_wrong = [r for r in valid if not r.get("initial_correct", True)]
        c2w_rates.append(
            sum(1 for r in initially_correct if not r["correct"]) / max(1, len(initially_correct))
        )
        w2c_rates.append(
            sum(1 for r in initially_wrong if r["correct"]) / max(1, len(initially_wrong))
        )
        avg_tokens.append(
            sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / max(1, len(valid))
        )

    # Print table
    print(f"\n{'Level':<20} {'Accuracy':>10} {'95% CI':>16} {'Changed':>10} {'C→W':>8} {'W→C':>8} {'Tokens':>10}")
    print("-" * 85)
    for i, level in enumerate(levels):
        print(f"{level:<20} {accuracies[i]:>10.1%} [{ci_low[i]:.3f}, {ci_high[i]:.3f}] {change_rates[i]:>10.1%} {c2w_rates[i]:>8.1%} {w2c_rates[i]:>8.1%} {avg_tokens[i]:>10.0f}")

    # Statistical tests vs baseline
    print("\nMcNemar's test (each level vs. L0_baseline):")
    for level in levels[1:]:
        test = mcnemar_test(data["L0_baseline"], data[level])
        h = cohens_h(accuracies[levels.index(level)], accuracies[0])
        print(f"  {level}: p={test['p_value']:.4f}, Cohen's h={h:.3f} (b={test['b']}, c={test['c']})")

    # Plot 1: Accuracy by critique level
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    short_labels = ["None", "Gentle", "Moderate", "Harsh", "Adversarial"]
    x = range(len(levels))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, [a * 100 for a in accuracies], color=colors, alpha=0.8)
    ax.errorbar(x, [a * 100 for a in accuracies],
                yerr=[[100*(a-l) for a, l in zip(accuracies, ci_low)],
                      [100*(h-a) for a, h in zip(accuracies, ci_high)]],
                fmt="none", color="black", capsize=5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Critique Stringency")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15)
    ax.set_ylim(0, 100)

    # Answer change dynamics
    ax = axes[1]
    width = 0.35
    x_arr = np.arange(len(levels))
    ax.bar(x_arr - width/2, [r * 100 for r in c2w_rates], width, label="Correct→Wrong", color="#F44336", alpha=0.7)
    ax.bar(x_arr + width/2, [r * 100 for r in w2c_rates], width, label="Wrong→Correct", color="#4CAF50", alpha=0.7)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Answer Change Dynamics")
    ax.set_xticks(x_arr)
    ax.set_xticklabels(short_labels, rotation=15)
    ax.legend()

    # Token usage
    ax = axes[2]
    ax.bar(x, avg_tokens, color=colors, alpha=0.8)
    ax.set_ylabel("Average Total Tokens")
    ax.set_title("Computational Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "experiment1_critique_stringency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'experiment1_critique_stringency.png'}")

    return {"accuracies": dict(zip(levels, accuracies)), "ci": dict(zip(levels, zip(ci_low, ci_high)))}


def analyze_experiment2(model_suffix: str = "gpt-4_1-mini"):
    """Analyze and visualize Experiment 2: Budget Control."""
    data = load_results(f"experiment2_budget_{model_suffix}.json")
    if not data:
        return None

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Budget Control")
    print("=" * 60)

    conditions = list(data.keys())
    accuracies = []
    ci_low = []
    ci_high = []
    avg_words = []
    avg_tokens = []

    for cond in conditions:
        results = data[cond]
        acc, lo, hi, n = compute_accuracy_ci(results)
        accuracies.append(acc)
        ci_low.append(lo)
        ci_high.append(hi)

        valid = [r for r in results if "error" not in r]
        avg_words.append(sum(r.get("response_length_words", 0) for r in valid) / max(1, len(valid)))
        avg_tokens.append(sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / max(1, len(valid)))

    # Print table
    print(f"\n{'Condition':<20} {'Accuracy':>10} {'95% CI':>16} {'Avg Words':>10} {'Avg Tokens':>12}")
    print("-" * 70)
    for i, cond in enumerate(conditions):
        print(f"{cond:<20} {accuracies[i]:>10.1%} [{ci_low[i]:.3f}, {ci_high[i]:.3f}] {avg_words[i]:>10.0f} {avg_tokens[i]:>12.0f}")

    # Statistical tests
    print("\nMcNemar's test (each budget vs. no_limit):")
    for cond in conditions[1:]:
        test = mcnemar_test(data["no_limit"], data[cond])
        h = cohens_h(accuracies[conditions.index(cond)], accuracies[0])
        print(f"  {cond}: p={test['p_value']:.4f}, Cohen's h={h:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    short_labels = ["No Limit", "300w", "150w", "75w"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    ax = axes[0]
    bars = ax.bar(range(len(conditions)), [a * 100 for a in accuracies], color=colors, alpha=0.8)
    ax.errorbar(range(len(conditions)), [a * 100 for a in accuracies],
                yerr=[[100*(a-l) for a, l in zip(accuracies, ci_low)],
                      [100*(h-a) for a, h in zip(accuracies, ci_high)]],
                fmt="none", color="black", capsize=5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Budget Constraint")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 100)

    ax = axes[1]
    ax.bar(range(len(conditions)), avg_words, color=colors, alpha=0.8)
    ax.set_ylabel("Average Response Length (words)")
    ax.set_title("Response Length by Budget")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(short_labels)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "experiment2_budget_control.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'experiment2_budget_control.png'}")

    return {"accuracies": dict(zip(conditions, accuracies))}


def analyze_experiment3(model_suffix: str = "gpt-4_1-mini"):
    """Analyze and visualize Experiment 3: Combined Critique × Budget."""
    data = load_results(f"experiment3_combined_{model_suffix}.json")
    if not data:
        return None

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Critique × Budget Interaction")
    print("=" * 60)

    # Parse conditions into a matrix
    critique_names = ["no_critique", "moderate_critique", "harsh_critique"]
    budget_names = ["no_limit", "moderate_budget", "tight_budget"]

    acc_matrix = np.zeros((len(critique_names), len(budget_names)))
    token_matrix = np.zeros((len(critique_names), len(budget_names)))

    print(f"\n{'Condition':<45} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-" * 70)
    for ci, critique in enumerate(critique_names):
        for bi, budget in enumerate(budget_names):
            cond = f"{critique}__{budget}"
            if cond not in data:
                continue
            results = data[cond]
            acc, _, _, n = compute_accuracy_ci(results)
            valid = [r for r in results if "error" not in r]
            avg_tok = sum(r.get("total_tokens", 0) for r in valid) / max(1, len(valid))
            acc_matrix[ci, bi] = acc * 100
            token_matrix[ci, bi] = avg_tok
            print(f"{cond:<45} {acc:>10.1%} {avg_tok:>12.0f}")

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    critique_labels = ["No Critique", "Moderate", "Harsh"]
    budget_labels = ["No Limit", "150w", "75w"]

    for ax, matrix, title, cmap in [
        (axes[0], acc_matrix, "Accuracy (%)", "RdYlGn"),
        (axes[1], token_matrix, "Avg Total Tokens", "YlOrRd"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(budget_labels)))
        ax.set_xticklabels(budget_labels)
        ax.set_yticks(range(len(critique_labels)))
        ax.set_yticklabels(critique_labels)
        ax.set_xlabel("Budget Constraint")
        ax.set_ylabel("Critique Level")
        ax.set_title(title)

        # Add text annotations
        for i in range(len(critique_labels)):
            for j in range(len(budget_labels)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                       color="black", fontweight="bold")

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "experiment3_combined_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'experiment3_combined_heatmap.png'}")

    return {"accuracy_matrix": acc_matrix.tolist()}


def analyze_experiment4(model_suffix: str = "gpt-4_1-mini"):
    """Analyze and visualize Experiment 4: Alternative Strategies."""
    data = load_results(f"experiment4_alternatives_{model_suffix}.json")
    if not data:
        return None

    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Alternative Effort-Inducing Strategies")
    print("=" * 60)

    strategies = list(data.keys())
    accuracies = []
    ci_low = []
    ci_high = []
    avg_tokens = []

    for strat in strategies:
        results = data[strat]
        acc, lo, hi, n = compute_accuracy_ci(results)
        accuracies.append(acc)
        ci_low.append(lo)
        ci_high.append(hi)
        valid = [r for r in results if "error" not in r]
        avg_tokens.append(sum(r.get("total_tokens", 0) for r in valid) / max(1, len(valid)))

    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'95% CI':>16} {'Avg Tokens':>12}")
    print("-" * 65)
    for i, strat in enumerate(strategies):
        print(f"{strat:<25} {accuracies[i]:>10.1%} [{ci_low[i]:.3f}, {ci_high[i]:.3f}] {avg_tokens[i]:>12.0f}")

    # Statistical tests vs baseline
    print("\nMcNemar's test (each strategy vs. baseline_cot):")
    for strat in strategies[1:]:
        test = mcnemar_test(data["baseline_cot"], data[strat])
        h = cohens_h(accuracies[strategies.index(strat)], accuracies[0])
        print(f"  {strat}: p={test['p_value']:.4f}, Cohen's h={h:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    short_labels = ["Baseline\nCoT", "High\nStakes", "Explicit\nRubric", "Step\nVerification", "Self-Consistency\n(k=5)"]
    colors = ["#2196F3", "#9C27B0", "#FF9800", "#4CAF50", "#F44336"]

    bars = ax.bar(range(len(strategies)), [a * 100 for a in accuracies], color=colors, alpha=0.8)
    ax.errorbar(range(len(strategies)), [a * 100 for a in accuracies],
                yerr=[[100*(a-l) for a, l in zip(accuracies, ci_low)],
                      [100*(h-a) for a, h in zip(accuracies, ci_high)]],
                fmt="none", color="black", capsize=5)

    # Add token cost as text on bars
    for i, (bar, tok) in enumerate(zip(bars, avg_tokens)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{tok:.0f} tok", ha="center", va="bottom", fontsize=9, color="gray")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Alternative Strategies for Improving LLM Output Quality")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "experiment4_alternative_strategies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'experiment4_alternative_strategies.png'}")

    return {"accuracies": dict(zip(strategies, accuracies))}


def create_summary_plot(model_suffix: str = "gpt-4_1-mini"):
    """Create a summary plot comparing the best strategy from each experiment."""
    exp1 = load_results(f"experiment1_critique_{model_suffix}.json")
    exp2 = load_results(f"experiment2_budget_{model_suffix}.json")
    exp3 = load_results(f"experiment3_combined_{model_suffix}.json")
    exp4 = load_results(f"experiment4_alternatives_{model_suffix}.json")

    if not all([exp1, exp2, exp3, exp4]):
        print("Not all experiment results available for summary plot")
        return

    # Collect best from each experiment
    conditions = {}

    # Baseline from exp1
    if "L0_baseline" in exp1:
        acc, lo, hi, _ = compute_accuracy_ci(exp1["L0_baseline"])
        valid = [r for r in exp1["L0_baseline"] if "error" not in r]
        tok = sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / max(1, len(valid))
        conditions["Baseline CoT"] = (acc, lo, hi, tok)

    # Best critique from exp1
    best_critique = None
    best_critique_acc = 0
    for level in ["L1_gentle", "L2_moderate", "L3_harsh", "L4_adversarial"]:
        if level in exp1:
            acc, lo, hi, _ = compute_accuracy_ci(exp1[level])
            if acc > best_critique_acc:
                best_critique_acc = acc
                valid = [r for r in exp1[level] if "error" not in r]
                tok = sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / max(1, len(valid))
                best_critique = (level, acc, lo, hi, tok)
    if best_critique:
        conditions[f"Best Critique\n({best_critique[0]})"] = best_critique[1:]

    # Best budget from exp2
    best_budget = None
    best_budget_acc = 0
    for cond in exp2:
        if cond == "no_limit":
            continue
        acc, lo, hi, _ = compute_accuracy_ci(exp2[cond])
        if acc > best_budget_acc:
            best_budget_acc = acc
            valid = [r for r in exp2[cond] if "error" not in r]
            tok = sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / max(1, len(valid))
            best_budget = (cond, acc, lo, hi, tok)
    if best_budget:
        conditions[f"Best Budget\n({best_budget[0]})"] = best_budget[1:]

    # Best combined from exp3
    best_combined = None
    best_combined_acc = 0
    for cond in exp3:
        if cond == "no_critique__no_limit":
            continue
        acc, lo, hi, _ = compute_accuracy_ci(exp3[cond])
        if acc > best_combined_acc:
            best_combined_acc = acc
            valid = [r for r in exp3[cond] if "error" not in r]
            tok = sum(r.get("total_tokens", 0) for r in valid) / max(1, len(valid))
            best_combined = (cond, acc, lo, hi, tok)
    if best_combined:
        conditions[f"Best Combined\n({best_combined[0].replace('__', '+')})"] = best_combined[1:]

    # Best alternative from exp4 (excluding baseline)
    best_alt = None
    best_alt_acc = 0
    for strat in exp4:
        if strat == "baseline_cot":
            continue
        acc, lo, hi, _ = compute_accuracy_ci(exp4[strat])
        if acc > best_alt_acc:
            best_alt_acc = acc
            valid = [r for r in exp4[strat] if "error" not in r]
            tok = sum(r.get("total_tokens", 0) for r in valid) / max(1, len(valid))
            best_alt = (strat, acc, lo, hi, tok)
    if best_alt:
        conditions[f"Best Alternative\n({best_alt[0]})"] = best_alt[1:]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    names = list(conditions.keys())
    accs = [conditions[n][0] * 100 for n in names]
    los = [100 * (conditions[n][0] - conditions[n][1]) for n in names]
    his = [100 * (conditions[n][2] - conditions[n][0]) for n in names]
    tokens = [conditions[n][3] for n in names]

    colors = ["#607D8B", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"][:len(names)]
    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85)
    ax.errorbar(range(len(names)), accs, yerr=[los, his], fmt="none", color="black", capsize=6)

    for i, (bar, tok) in enumerate(zip(bars, tokens)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{tok:.0f} tokens", ha="center", va="bottom", fontsize=9, color="gray")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Best Strategy from Each Experiment Category\n(GSM8K, GPT-4.1-mini)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(y=accs[0], color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_best_strategies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary plot saved to {PLOTS_DIR / 'summary_best_strategies.png'}")


def run_all_analyses(model_suffix: str = "gpt-4_1-mini"):
    """Run all analyses."""
    r1 = analyze_experiment1(model_suffix)
    r2 = analyze_experiment2(model_suffix)
    r3 = analyze_experiment3(model_suffix)
    r4 = analyze_experiment4(model_suffix)
    create_summary_plot(model_suffix)

    # Save combined analysis
    summary = {
        "experiment1": r1,
        "experiment2": r2,
        "experiment3": r3,
        "experiment4": r4,
    }
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAnalysis summary saved to {RESULTS_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-suffix", type=str, default="gpt-4_1-mini")
    args = parser.parse_args()
    run_all_analyses(args.model_suffix)
