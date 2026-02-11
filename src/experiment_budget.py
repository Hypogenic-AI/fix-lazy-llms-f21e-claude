"""Experiment 2: Budget Control on GSM8K.

Tests whether constraining response length forces models to be more
efficient and accurate in their reasoning.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_gsm8k, call_llm, extract_numerical_answer, check_answer, RESULTS_DIR
from tqdm import tqdm

BUDGET_CONDITIONS = {
    "no_limit": None,
    "generous_300w": "Keep your solution to at most 300 words.",
    "moderate_150w": "Keep your solution to at most 150 words. Be concise.",
    "tight_75w": "Keep your solution to at most 75 words. Be very concise — only essential steps.",
}

SYSTEM_PROMPT = "You are a helpful math assistant. Show your work and end with your final numerical answer on a line like: #### [number]"


def run_experiment(n_problems: int = 200, model: str = "gpt-4.1-mini"):
    """Run the budget control experiment."""
    print(f"=== Experiment 2: Budget Control ===")
    print(f"Model: {model}, Problems: {n_problems}")

    problems = load_gsm8k("test", n=n_problems)
    results = {cond: [] for cond in BUDGET_CONDITIONS}

    for cond, budget_instruction in BUDGET_CONDITIONS.items():
        print(f"\n--- Running {cond} ---")
        for i, problem in enumerate(tqdm(problems, desc=cond)):
            try:
                prompt = f"Solve this math problem step by step:\n\n{problem['question']}"
                if budget_instruction:
                    prompt += f"\n\n{budget_instruction}"

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                result = call_llm(messages, model=model)
                answer = extract_numerical_answer(result["content"])
                correct = check_answer(answer, problem["numerical_answer"])

                results[cond].append({
                    "response": result["content"],
                    "extracted_answer": answer,
                    "gold_answer": problem["numerical_answer"],
                    "correct": correct,
                    "usage": result["usage"],
                    "response_length_words": len(result["content"].split()),
                })
            except Exception as e:
                print(f"  Error on problem {i}: {e}")
                results[cond].append({"error": str(e), "correct": False})

    # Save
    output_path = RESULTS_DIR / f"experiment2_budget_{model.replace('.', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Condition':<20} {'Accuracy':>10} {'Avg Words':>10} {'Avg Tokens':>12}")
    print("-" * 55)
    for cond, cond_results in results.items():
        valid = [r for r in cond_results if "error" not in r]
        if not valid:
            continue
        accuracy = sum(1 for r in valid if r["correct"]) / len(valid)
        avg_words = sum(r["response_length_words"] for r in valid) / len(valid)
        avg_tokens = sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / len(valid)
        print(f"{cond:<20} {accuracy:>10.1%} {avg_words:>10.0f} {avg_tokens:>12.0f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()
    run_experiment(n_problems=args.n, model=args.model)
