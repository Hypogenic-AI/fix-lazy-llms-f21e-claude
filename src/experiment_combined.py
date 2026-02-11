"""Experiment 3: Critique × Budget Interaction (Factorial Design).

Tests whether combining critique stringency with budget control
produces better results than either alone.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_gsm8k, call_llm, extract_numerical_answer, check_answer, RESULTS_DIR
from tqdm import tqdm

CRITIQUE_LEVELS = {
    "no_critique": None,
    "moderate_critique": (
        "Carefully examine each step. Identify any logical errors, calculation mistakes, "
        "or unjustified leaps. If you find errors, correct them. Provide your final answer."
    ),
    "harsh_critique": (
        "You are an extremely strict math professor. Scrutinize every step. "
        "Find every flaw — wrong arithmetic, logical gaps, incorrect assumptions. "
        "Assume there ARE errors. After critique, provide the corrected final answer."
    ),
}

BUDGET_LEVELS = {
    "no_limit": None,
    "moderate_budget": "Keep your response to at most 150 words. Be concise.",
    "tight_budget": "Keep your response to at most 75 words. Only essential steps.",
}

SYSTEM_PROMPT = "You are a helpful math assistant. Show your work and end with your final numerical answer on a line like: #### [number]"


def run_experiment(n_problems: int = 200, model: str = "gpt-4.1-mini"):
    """Run the combined critique × budget factorial experiment."""
    print(f"=== Experiment 3: Critique × Budget Factorial ===")
    print(f"Model: {model}, Problems: {n_problems}")

    problems = load_gsm8k("test", n=n_problems)
    results = {}

    for critique_name, critique_prompt in CRITIQUE_LEVELS.items():
        for budget_name, budget_instruction in BUDGET_LEVELS.items():
            condition = f"{critique_name}__{budget_name}"
            results[condition] = []
            print(f"\n--- Running {condition} ---")

            for i, problem in enumerate(tqdm(problems, desc=condition)):
                try:
                    # Build the initial prompt
                    user_prompt = f"Solve this math problem step by step:\n\n{problem['question']}"
                    if budget_instruction and critique_prompt is None:
                        user_prompt += f"\n\n{budget_instruction}"

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]
                    gen_result = call_llm(messages, model=model)
                    initial_response = gen_result["content"]
                    initial_answer = extract_numerical_answer(initial_response)

                    if critique_prompt is not None:
                        # Apply critique
                        critique_text = critique_prompt
                        if budget_instruction:
                            critique_text += f"\n\n{budget_instruction}"
                        critique_text += "\n\nEnd with your final answer on a line like: #### [number]"

                        critique_messages = messages + [
                            {"role": "assistant", "content": initial_response},
                            {"role": "user", "content": critique_text},
                        ]
                        critique_result = call_llm(critique_messages, model=model)
                        final_response = critique_result["content"]
                        final_answer = extract_numerical_answer(final_response)
                        total_tokens = gen_result["usage"]["total_tokens"] + critique_result["usage"]["total_tokens"]
                    else:
                        final_response = initial_response
                        final_answer = initial_answer
                        total_tokens = gen_result["usage"]["total_tokens"]

                    correct = check_answer(final_answer, problem["numerical_answer"])
                    initial_correct = check_answer(initial_answer, problem["numerical_answer"])

                    results[condition].append({
                        "final_answer": final_answer,
                        "initial_answer": initial_answer,
                        "gold_answer": problem["numerical_answer"],
                        "correct": correct,
                        "initial_correct": initial_correct,
                        "answer_changed": initial_answer != final_answer,
                        "total_tokens": total_tokens,
                        "response_length_words": len(final_response.split()),
                    })
                except Exception as e:
                    print(f"  Error on problem {i}: {e}")
                    results[condition].append({"error": str(e), "correct": False})

    # Save
    output_path = RESULTS_DIR / f"experiment3_combined_{model.replace('.', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Condition':<40} {'Accuracy':>10} {'Avg Words':>10} {'Avg Tokens':>12}")
    print("-" * 75)
    for cond, cond_results in results.items():
        valid = [r for r in cond_results if "error" not in r]
        if not valid:
            continue
        accuracy = sum(1 for r in valid if r["correct"]) / len(valid)
        avg_words = sum(r["response_length_words"] for r in valid) / len(valid)
        avg_tokens = sum(r["total_tokens"] for r in valid) / len(valid)
        print(f"{cond:<40} {accuracy:>10.1%} {avg_words:>10.0f} {avg_tokens:>12.0f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()
    run_experiment(n_problems=args.n, model=args.model)
