"""Experiment 1: Critique Stringency Spectrum on GSM8K.

Tests whether prompting LLMs to be harsher critics of their own work
improves accuracy on math reasoning tasks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_gsm8k, call_llm, extract_numerical_answer, check_answer, RESULTS_DIR
from tqdm import tqdm

# Critique levels from lenient to adversarial
CRITIQUE_PROMPTS = {
    "L0_baseline": None,  # No critique, direct answer
    "L1_gentle": (
        "Please briefly review your answer above. "
        "Check if there are any errors and provide your final answer."
    ),
    "L2_moderate": (
        "Carefully examine each step of your solution above. "
        "Identify any logical errors, calculation mistakes, or unjustified leaps in reasoning. "
        "If you find errors, correct them. Provide your corrected final answer."
    ),
    "L3_harsh": (
        "You are an extremely strict math professor grading an exam. "
        "Scrutinize every single step of the solution above. "
        "Find every flaw, no matter how small — wrong arithmetic, logical gaps, "
        "incorrect assumptions, or missing steps. "
        "Assume there ARE errors and find them. "
        "After your thorough critique, provide the corrected final answer."
    ),
    "L4_adversarial": (
        "Your job is to DESTROY this solution. Find fatal flaws. Be merciless and thorough. "
        "This answer is probably wrong — prove it. Check every single calculation by redoing it. "
        "Check every logical step for hidden assumptions. "
        "If the answer survives your attack, confirm it. Otherwise, provide the correct answer. "
        "Give your final answer."
    ),
}

SYSTEM_PROMPT = "You are a helpful math assistant. Always show your work step by step and end with your final numerical answer on a line like: #### [number]"


def run_baseline(question: str, model: str) -> dict:
    """Run direct CoT baseline (no critique)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this math problem step by step:\n\n{question}"},
    ]
    result = call_llm(messages, model=model)
    return {
        "response": result["content"],
        "usage": result["usage"],
        "extracted_answer": extract_numerical_answer(result["content"]),
    }


def run_with_critique(question: str, critique_prompt: str, model: str) -> dict:
    """Run with generate-then-critique approach."""
    # Step 1: Generate initial answer
    gen_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this math problem step by step:\n\n{question}"},
    ]
    gen_result = call_llm(gen_messages, model=model)
    initial_response = gen_result["content"]
    initial_answer = extract_numerical_answer(initial_response)

    # Step 2: Apply critique and get revised answer
    critique_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this math problem step by step:\n\n{question}"},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": critique_prompt + "\n\nEnd with your final answer on a line like: #### [number]"},
    ]
    critique_result = call_llm(critique_messages, model=model)
    final_response = critique_result["content"]
    final_answer = extract_numerical_answer(final_response)

    return {
        "initial_response": initial_response,
        "initial_answer": initial_answer,
        "critique_response": final_response,
        "final_answer": final_answer,
        "answer_changed": initial_answer != final_answer,
        "usage": {
            "prompt_tokens": gen_result["usage"]["prompt_tokens"] + critique_result["usage"]["prompt_tokens"],
            "completion_tokens": gen_result["usage"]["completion_tokens"] + critique_result["usage"]["completion_tokens"],
            "total_tokens": gen_result["usage"]["total_tokens"] + critique_result["usage"]["total_tokens"],
        },
    }


def run_experiment(n_problems: int = 200, model: str = "gpt-4.1-mini"):
    """Run the full critique stringency experiment."""
    print(f"=== Experiment 1: Critique Stringency Spectrum ===")
    print(f"Model: {model}, Problems: {n_problems}")

    problems = load_gsm8k("test", n=n_problems)
    results = {level: [] for level in CRITIQUE_PROMPTS}

    for level, critique_prompt in CRITIQUE_PROMPTS.items():
        print(f"\n--- Running {level} ---")
        for i, problem in enumerate(tqdm(problems, desc=level)):
            try:
                if critique_prompt is None:
                    res = run_baseline(problem["question"], model)
                    res["final_answer"] = res["extracted_answer"]
                    res["answer_changed"] = False
                else:
                    res = run_with_critique(problem["question"], critique_prompt, model)

                res["gold_answer"] = problem["numerical_answer"]
                res["correct"] = check_answer(res["final_answer"], problem["numerical_answer"])

                # For critique levels, also check if initial answer was correct
                if critique_prompt is not None:
                    res["initial_correct"] = check_answer(res.get("initial_answer"), problem["numerical_answer"])
                else:
                    res["initial_correct"] = res["correct"]

                results[level].append(res)
            except Exception as e:
                print(f"  Error on problem {i}: {e}")
                results[level].append({"error": str(e), "gold_answer": problem["numerical_answer"], "correct": False})

    # Save results
    output_path = RESULTS_DIR / f"experiment1_critique_{model.replace('.', '_')}.json"
    # Convert for JSON serialization
    save_data = {}
    for level, level_results in results.items():
        save_data[level] = []
        for r in level_results:
            save_data[level].append({k: v for k, v in r.items()})

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Level':<20} {'Accuracy':>10} {'Ans Changed':>12} {'Correct→Wrong':>14} {'Wrong→Correct':>14} {'Avg Tokens':>12}")
    print("-" * 85)
    for level, level_results in results.items():
        valid = [r for r in level_results if "error" not in r]
        if not valid:
            continue
        accuracy = sum(1 for r in valid if r["correct"]) / len(valid)
        changed = sum(1 for r in valid if r.get("answer_changed", False)) / len(valid)

        # Correct→Wrong and Wrong→Correct rates
        c2w = sum(1 for r in valid if r.get("initial_correct") and not r["correct"]) / max(1, sum(1 for r in valid if r.get("initial_correct")))
        w2c = sum(1 for r in valid if not r.get("initial_correct") and r["correct"]) / max(1, sum(1 for r in valid if not r.get("initial_correct")))

        avg_tokens = sum(r["usage"]["total_tokens"] for r in valid if "usage" in r) / len(valid)
        print(f"{level:<20} {accuracy:>10.1%} {changed:>12.1%} {c2w:>14.1%} {w2c:>14.1%} {avg_tokens:>12.0f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()
    run_experiment(n_problems=args.n, model=args.model)
