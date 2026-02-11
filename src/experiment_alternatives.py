"""Experiment 4: Alternative Effort-Inducing Strategies.

Tests whether approaches other than critique (high-stakes framing,
explicit rubrics, step verification, self-consistency) improve output quality.
"""

import json
import sys
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_gsm8k, call_llm, extract_numerical_answer, check_answer, RESULTS_DIR, SEED
from tqdm import tqdm

SYSTEM_PROMPT = "You are a helpful math assistant. Show your work and end with your final numerical answer on a line like: #### [number]"

STRATEGIES = {
    "baseline_cot": {
        "description": "Standard chain-of-thought",
        "type": "single",
        "prompt": "Solve this math problem step by step:\n\n{question}",
    },
    "high_stakes": {
        "description": "Emotional/urgency framing",
        "type": "single",
        "prompt": (
            "This is an extremely important exam that will determine your entire career. "
            "You MUST get this right. Take your time, be extremely careful, and double-check everything. "
            "Solve this math problem step by step:\n\n{question}"
        ),
    },
    "explicit_rubric": {
        "description": "Provide scoring rubric",
        "type": "critique",
        "initial_prompt": "Solve this math problem step by step:\n\n{question}",
        "critique_prompt": (
            "Now evaluate your solution against this rubric:\n"
            "1. Are all quantities from the problem correctly identified? (Yes/No)\n"
            "2. Is each arithmetic operation correct? Verify by computing each one.\n"
            "3. Does the final answer have correct units and make sense given the problem?\n"
            "4. Are there any logical leaps or missing steps?\n\n"
            "Score each criterion, then provide your corrected final answer.\n"
            "End with: #### [number]"
        ),
    },
    "step_verification": {
        "description": "Verify by re-computing each step",
        "type": "critique",
        "initial_prompt": "Solve this math problem step by step:\n\n{question}",
        "critique_prompt": (
            "Now verify your solution by working through it again from scratch, independently. "
            "Do NOT refer to your previous work — solve the problem fresh. "
            "If you get a different answer, carefully determine which one is correct.\n"
            "End with your final answer on a line like: #### [number]"
        ),
    },
    "self_consistency_5": {
        "description": "Majority vote over 5 samples (temp=0.7)",
        "type": "self_consistency",
        "prompt": "Solve this math problem step by step:\n\n{question}",
        "n_samples": 5,
        "temperature": 0.7,
    },
}


def run_experiment(n_problems: int = 200, model: str = "gpt-4.1-mini"):
    """Run the alternative strategies experiment."""
    print(f"=== Experiment 4: Alternative Effort-Inducing Strategies ===")
    print(f"Model: {model}, Problems: {n_problems}")

    problems = load_gsm8k("test", n=n_problems)
    results = {name: [] for name in STRATEGIES}

    for strat_name, strat in STRATEGIES.items():
        print(f"\n--- Running {strat_name}: {strat['description']} ---")

        for i, problem in enumerate(tqdm(problems, desc=strat_name)):
            try:
                if strat["type"] == "single":
                    prompt = strat["prompt"].format(question=problem["question"])
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    result = call_llm(messages, model=model)
                    answer = extract_numerical_answer(result["content"])
                    total_tokens = result["usage"]["total_tokens"]

                    results[strat_name].append({
                        "final_answer": answer,
                        "gold_answer": problem["numerical_answer"],
                        "correct": check_answer(answer, problem["numerical_answer"]),
                        "total_tokens": total_tokens,
                        "response_length_words": len(result["content"].split()),
                    })

                elif strat["type"] == "critique":
                    # Step 1: Generate
                    init_prompt = strat["initial_prompt"].format(question=problem["question"])
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": init_prompt},
                    ]
                    gen_result = call_llm(messages, model=model)
                    initial_answer = extract_numerical_answer(gen_result["content"])

                    # Step 2: Critique
                    critique_messages = messages + [
                        {"role": "assistant", "content": gen_result["content"]},
                        {"role": "user", "content": strat["critique_prompt"]},
                    ]
                    critique_result = call_llm(critique_messages, model=model)
                    final_answer = extract_numerical_answer(critique_result["content"])
                    total_tokens = gen_result["usage"]["total_tokens"] + critique_result["usage"]["total_tokens"]

                    results[strat_name].append({
                        "initial_answer": initial_answer,
                        "final_answer": final_answer,
                        "gold_answer": problem["numerical_answer"],
                        "correct": check_answer(final_answer, problem["numerical_answer"]),
                        "initial_correct": check_answer(initial_answer, problem["numerical_answer"]),
                        "answer_changed": initial_answer != final_answer,
                        "total_tokens": total_tokens,
                        "response_length_words": len(critique_result["content"].split()),
                    })

                elif strat["type"] == "self_consistency":
                    prompt = strat["prompt"].format(question=problem["question"])
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    answers = []
                    total_tokens = 0
                    for _ in range(strat["n_samples"]):
                        result = call_llm(
                            messages, model=model,
                            temperature=strat["temperature"],
                            use_cache=False,  # Need different samples
                        )
                        ans = extract_numerical_answer(result["content"])
                        answers.append(ans)
                        total_tokens += result["usage"]["total_tokens"]

                    # Majority vote
                    answer_counts = Counter(a for a in answers if a is not None)
                    if answer_counts:
                        majority_answer = answer_counts.most_common(1)[0][0]
                    else:
                        majority_answer = None

                    results[strat_name].append({
                        "all_answers": answers,
                        "final_answer": majority_answer,
                        "gold_answer": problem["numerical_answer"],
                        "correct": check_answer(majority_answer, problem["numerical_answer"]),
                        "total_tokens": total_tokens,
                        "agreement_rate": max(answer_counts.values()) / len(answers) if answer_counts else 0,
                    })

            except Exception as e:
                print(f"  Error on problem {i}: {e}")
                results[strat_name].append({"error": str(e), "correct": False})

    # Save
    output_path = RESULTS_DIR / f"experiment4_alternatives_{model.replace('.', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Strategy':<25} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-" * 50)
    for strat_name, strat_results in results.items():
        valid = [r for r in strat_results if "error" not in r]
        if not valid:
            continue
        accuracy = sum(1 for r in valid if r["correct"]) / len(valid)
        avg_tokens = sum(r["total_tokens"] for r in valid) / len(valid)
        print(f"{strat_name:<25} {accuracy:>10.1%} {avg_tokens:>12.0f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()
    run_experiment(n_problems=args.n, model=args.model)
