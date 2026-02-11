# Fixing Lazy LLMs

**Does prompting LLMs to be harsher critics improve output quality?**

An empirical investigation of critique stringency, budget control, and alternative effort-inducing strategies on the GSM8K math reasoning benchmark.

## Key Findings

| Intervention | Accuracy | Tokens | Verdict |
|-------------|----------|--------|---------|
| Baseline chain-of-thought | 96.0% | 305 | Best accuracy |
| Harsh self-critique | 91.0% | 1,218 | Hurts accuracy, wastes tokens |
| Budget constraint (75w) | 96.7% | 198 | Best efficiency |
| Self-consistency (k=5) | 95.3% | 1,454 | No gain, 5× cost |

**Bottom line**: Don't ask LLMs to critique themselves on reasoning tasks — it makes them worse. Instead, ask them to be concise.

## Experiments

1. **Critique Stringency** (Exp 1): 5 levels from none to adversarial on 200 GSM8K problems
2. **Budget Control** (Exp 2): 4 budget constraints on 150 problems
3. **Critique × Budget Factorial** (Exp 3): 3×3 design on 150 problems
4. **Alternative Strategies** (Exp 4): High-stakes framing, rubrics, step verification, self-consistency on 150 problems

See [REPORT.md](REPORT.md) for full results, statistical analysis, and discussion.

## Project Structure

```
src/
  utils.py                    # Shared utilities (API calls, caching, answer extraction)
  experiment_critique.py      # Experiment 1: Critique stringency spectrum
  experiment_budget.py        # Experiment 2: Budget control
  experiment_combined.py      # Experiment 3: Critique × Budget factorial
  experiment_alternatives.py  # Experiment 4: Alternative strategies
  analysis.py                 # Statistical analysis and visualization
  run_all.py                  # Master orchestration script
datasets/
  gsm8k/test.jsonl            # GSM8K test set (1,319 problems)
results/
  experiment1_critique_*.json # Raw results
  experiment2_budget_*.json
  experiment3_combined_*.json
  experiment4_alternatives_*.json
  analysis_summary.json       # Aggregated analysis
  plots/                      # Generated visualizations
planning.md                   # Research plan
literature_review.md          # Literature review (27 papers)
REPORT.md                     # Full research report
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai numpy scipy matplotlib tqdm
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Running Experiments

Run all experiments:
```bash
python src/run_all.py --n 200 --model gpt-4.1-mini
```

Run individual experiments:
```bash
python src/experiment_critique.py --n 200 --model gpt-4.1-mini
python src/experiment_budget.py --n 150 --model gpt-4.1-mini
python src/experiment_combined.py --n 150 --model gpt-4.1-mini
python src/experiment_alternatives.py --n 150 --model gpt-4.1-mini
```

Run analysis:
```bash
python src/analysis.py
```

## Model

All experiments use **GPT-4.1-mini** via the OpenAI API. API responses are cached (SHA-256 hash key) for reproducibility.

## Statistical Methods

- McNemar's test for paired binary outcomes
- Wilson score 95% confidence intervals
- Cohen's h effect size for proportion differences
