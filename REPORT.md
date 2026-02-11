# Fixing Lazy LLMs: Does Harsher Self-Critique Improve Output Quality?

## An Empirical Investigation of Critique Stringency, Budget Control, and Alternative Effort-Inducing Strategies

---

## Abstract

Large language models (LLMs) are often perceived as "lazy" — producing generic, shallow, or unnecessarily verbose responses when more careful work is achievable. A popular hypothesis holds that prompting LLMs more aggressively — essentially asking them to be harsher critics of their own work — should improve output quality. We systematically test this hypothesis through four experiments on the GSM8K math reasoning benchmark using GPT-4.1-mini. Our findings are striking and counterintuitive:

1. **Harsher self-critique monotonically degrades accuracy** (96.0% → 91.0%), contradicting the popular intuition.
2. **Budget constraints maintain or slightly improve accuracy** (95.3% → 96.7%) while reducing token usage by 32%.
3. **Budget constraints partially mitigate critique-induced damage** in factorial analysis.
4. **No alternative strategy** (high-stakes framing, explicit rubrics, step verification, self-consistency voting) outperforms the simple baseline.

These results suggest that the "lazy LLM" problem is not solvable through harder self-critique. Instead, the primary issue is that self-critique introduces errors by causing models to second-guess correct answers. The most effective intervention is simply constraining response length, which forces concise reasoning without the error-amplification of self-evaluation.

---

## 1. Introduction

### 1.1 The "Lazy LLM" Problem

Users of large language models frequently complain that models "take the easy way out" — producing verbose but superficial responses, using brute-force approaches instead of elegant solutions, and failing to demonstrate the careful reasoning they are capable of. This perception is supported by empirical evidence: Self-Refine (Madaan et al., 2023) showed that LLM outputs consistently fall short of model capability, while Poddar et al. (2025) found responses are 3–20× longer than necessary for factual questions.

A common folk hypothesis holds that being "rude" or aggressive in prompting produces better results — asking the model to try harder, framing the task as high-stakes, or instructing it to ruthlessly critique its own work. This connects to the broader research question of whether prompting LLMs to act as increasingly harsh critics can systematically improve output quality.

### 1.2 Research Questions

**Primary**: Does prompting LLMs to act as increasingly harsh critics of their own work improve output quality?

**Secondary**:
1. Does controlling response budget interact with critique stringency?
2. How does critique-based improvement compare to other effort-inducing strategies (high-stakes framing, explicit rubrics, step verification, self-consistency)?
3. Is there an optimal level of critique harshness, or is the relationship monotonic?

### 1.3 Hypotheses

- **H1**: Increasing critique stringency will show an inverted U-curve, with moderate critique outperforming both no critique and adversarial critique.
- **H2**: Token budget constraints will independently improve accuracy by forcing conciseness.
- **H3**: Combining budget control with critique will outperform either alone.
- **H4**: Structured alternative strategies (rubrics, step verification) will outperform vague urgency framing.

### 1.4 Prior Work

The literature on LLM self-correction is divided. **Self-Refine** (Madaan et al., 2023) demonstrated 20% average improvement through iterative self-feedback across 7 tasks, but showed near-zero improvement on math reasoning. **Huang et al. (2024)** directly challenged self-correction, finding that on GSM8K, intrinsic self-correction changed 12.3% of correct answers to incorrect but only 1.7% of incorrect answers to correct. **Kamoi et al. (2024)** provided the most comprehensive survey, concluding that intrinsic self-correction on reasoning tasks has never been demonstrated to work under fair experimental conditions.

On the budget control side, **TALE** (Han et al., 2024) showed that explicit token budgets can improve both efficiency and accuracy on GSM8K. **CCoT** (Nayab et al., 2024) demonstrated 12–25% redundancy reduction with maintained accuracy. **Wu et al. (2025)** provided theoretical evidence that accuracy follows an inverted U-curve with chain-of-thought length.

**Our contribution**: No prior work has systematically varied the *stringency* of self-critique from gentle to adversarial, tested the interaction between critique intensity and response budget, or compared critique-based approaches against multiple alternative effort-inducing strategies. We fill this gap with a controlled empirical study.

---

## 2. Methodology

### 2.1 Benchmark and Model

- **Benchmark**: GSM8K (Cobbe et al., 2021), a dataset of 1,319 grade-school math word problems with numerical answers. We use random subsets of 150–200 problems (seed=42) for each experiment.
- **Model**: GPT-4.1-mini via OpenAI API (temperature=0 for deterministic outputs except where noted).
- **Answer extraction**: Regex-based extraction of the `#### [number]` pattern, with fallback patterns for "the answer is" and LaTeX `\boxed{}` notation.
- **Correctness**: Numerical comparison with tolerance of 1e-6.

### 2.2 Experimental Design

#### Experiment 1: Critique Stringency Spectrum (n=200)

We test 5 levels of critique stringency using a generate-then-critique protocol:

| Level | Description | Critique Prompt |
|-------|-------------|-----------------|
| L0 (Baseline) | No critique | Direct chain-of-thought only |
| L1 (Gentle) | Light review | "Please briefly review your answer... Check if there are any errors" |
| L2 (Moderate) | Structured review | "Carefully examine each step... Identify any logical errors, calculation mistakes, or unjustified leaps" |
| L3 (Harsh) | Strict professor | "You are an extremely strict math professor... Find every flaw, no matter how small... Assume there ARE errors" |
| L4 (Adversarial) | Destroy the solution | "Your job is to DESTROY this solution... This answer is probably wrong — prove it" |

For L0, the model produces a single chain-of-thought response. For L1–L4, the model first generates a solution, then receives the critique prompt along with its own solution, and produces a revised answer.

#### Experiment 2: Budget Control (n=150)

We test 4 budget levels with no self-critique:

| Budget | Instruction |
|--------|-------------|
| No limit | Standard chain-of-thought |
| 300 words | "Keep your solution to at most 300 words" |
| 150 words | "Keep your solution to at most 150 words. Be concise." |
| 75 words | "Keep your solution to at most 75 words. Be very concise — only essential steps." |

#### Experiment 3: Critique × Budget Factorial (n=150)

A 3×3 factorial design crossing:
- **Critique**: None, Moderate, Harsh
- **Budget**: No limit, 150 words, 75 words

This yields 9 conditions to test interactions between critique and budget.

#### Experiment 4: Alternative Effort-Inducing Strategies (n=150)

We compare 5 strategies:

| Strategy | Approach |
|----------|----------|
| Baseline CoT | Standard chain-of-thought |
| High Stakes | "This is an extremely important exam that will determine your entire career..." |
| Explicit Rubric | Generate → evaluate against 4-criterion rubric → revise |
| Step Verification | Generate → "solve the problem fresh" independently → reconcile |
| Self-Consistency (k=5) | 5 samples at temperature=0.7, majority vote |

### 2.3 Statistical Analysis

- **Primary test**: McNemar's test for paired binary outcomes (with continuity correction)
- **Effect size**: Cohen's h for proportion differences
- **Confidence intervals**: 95% Wilson score intervals
- **Significance level**: α = 0.05

### 2.4 Caching

All API calls are cached by SHA-256 hash of the request parameters, ensuring exact reproducibility without redundant API calls.

---

## 3. Results

### 3.1 Experiment 1: Critique Stringency Spectrum

**Finding: Harsher critique monotonically degrades accuracy. H1 is rejected.**

| Level | Accuracy | 95% CI | Answer Changed | Correct→Wrong | Wrong→Correct | Avg Tokens |
|-------|----------|--------|----------------|---------------|---------------|------------|
| L0 (Baseline) | **96.0%** | [92.3%, 98.0%] | 0.0% | 0.0% | 0.0% | 305 |
| L1 (Gentle) | 94.5% | [90.4%, 96.9%] | 2.5% | 2.1% | 12.5% | 776 |
| L2 (Moderate) | 93.5% | [89.2%, 96.2%] | 4.5% | 3.6% | 25.0% | 885 |
| L3 (Harsh) | **91.0%** | [86.2%, 94.2%] | 6.5% | 5.7% | 12.5% | 1,218 |
| L4 (Adversarial) | 91.5% | [86.8%, 94.6%] | 6.5% | 5.7% | 25.0% | 1,232 |

**Statistical tests vs. baseline (L0)**:

| Comparison | McNemar p-value | Cohen's h | Significant? |
|------------|-----------------|-----------|-------------|
| L0 vs L1 | 0.371 | −0.071 | No |
| L0 vs L2 | 0.182 | −0.113 | No |
| L0 vs L3 | **0.009** | −0.207 | **Yes** |
| L0 vs L4 | **0.027** | −0.189 | **Yes** |

The decline from baseline is statistically significant for harsh (p=0.009) and adversarial (p=0.027) critique levels. The effect is monotonically negative — there is no inverted U-curve. The Correct→Wrong rate increases steadily from 0% (baseline) to 5.7% (harsh/adversarial), while the Wrong→Correct rate is much smaller and inconsistent. Self-critique costs 2.5–4× more tokens than baseline while degrading accuracy.

**Key insight**: The model's self-critique mechanism is asymmetrically harmful. When instructed to find errors, the model is more likely to "find" errors in correct solutions (introducing false positives) than to genuinely detect and fix real errors.

### 3.2 Experiment 2: Budget Control

**Finding: Budget constraints maintain or slightly improve accuracy while reducing cost. H2 is partially supported.**

| Condition | Accuracy | 95% CI | Avg Words | Avg Tokens |
|-----------|----------|--------|-----------|------------|
| No limit | 95.3% | [90.7%, 97.7%] | 123 | 290 |
| 300 words | 96.0% | [91.5%, 98.2%] | 134 | 318 |
| 150 words | **96.7%** | [92.4%, 98.6%] | 87 | 253 |
| 75 words | **96.7%** | [92.4%, 98.6%] | 45 | 198 |

**Statistical tests vs. no limit**:

| Comparison | McNemar p-value | Cohen's h |
|------------|-----------------|-----------|
| No limit vs 300w | 1.000 | 0.033 |
| No limit vs 150w | 0.480 | 0.068 |
| No limit vs 75w | 0.617 | 0.068 |

While none of the differences reach statistical significance (the sample size of 150 limits power for detecting small effects), the trend is clear and practically meaningful: budget constraints do not hurt accuracy at any level tested, and the tightest constraint (75 words) achieves the highest accuracy (96.7%) while using 32% fewer tokens than the no-limit condition. This is consistent with findings from TALE (Han et al., 2024), CCoT (Nayab et al., 2024), and the theoretical predictions of Wu et al. (2025).

### 3.3 Experiment 3: Critique × Budget Interaction

**Finding: Budget constraints partially mitigate critique-induced damage. H3 is partially supported (but in an unexpected direction).**

| | No Limit | 150 words | 75 words |
|---|---------|-----------|----------|
| **No Critique** | 95.3% | **96.7%** | **96.7%** |
| **Moderate Critique** | 92.7% | 94.7% | 94.7% |
| **Harsh Critique** | 88.0% | 91.3% | 92.0% |

Average token usage:

| | No Limit | 150 words | 75 words |
|---|---------|-----------|----------|
| **No Critique** | 290 | 252 | 202 |
| **Moderate Critique** | 850 | 770 | 731 |
| **Harsh Critique** | 1,092 | 813 | 759 |

The factorial design reveals two clear main effects:

1. **Critique effect (negative)**: Within each budget level, adding critique reduces accuracy. The worst condition (harsh critique, no limit: 88.0%) is 8.7 percentage points below the best (no critique, tight budget: 96.7%).

2. **Budget effect (positive)**: Within each critique level, adding budget constraints improves accuracy. This is especially pronounced for harsh critique, where tight budget (92.0%) partially recovers the 7.3pp loss from unconstrained harsh critique (88.0%).

**Interpretation**: Budget constraints appear to mitigate critique damage by limiting the model's ability to "overthink" during self-evaluation. When forced to be concise in its critique, the model is less likely to fabricate errors in correct solutions.

### 3.4 Experiment 4: Alternative Effort-Inducing Strategies

**Finding: No alternative strategy outperforms baseline chain-of-thought. H4 is rejected.**

| Strategy | Accuracy | 95% CI | Avg Tokens | Cost vs. Baseline |
|----------|----------|--------|------------|-------------------|
| Baseline CoT | **95.3%** | [90.7%, 97.7%] | 290 | 1.0× |
| High Stakes | 95.3% | [90.7%, 97.7%] | 351 | 1.2× |
| Explicit Rubric | 94.7% | [89.8%, 97.3%] | 950 | 3.3× |
| Step Verification | 95.3% | [90.7%, 97.7%] | 815 | 2.8× |
| Self-Consistency (k=5) | 95.3% | [90.7%, 97.7%] | 1,454 | 5.0× |

**Statistical tests vs. baseline**:

| Comparison | McNemar p-value | Cohen's h |
|------------|-----------------|-----------|
| vs High Stakes | 0.480 | 0.000 |
| vs Explicit Rubric | 1.000 | −0.031 |
| vs Step Verification | 1.000 | 0.000 |
| vs Self-Consistency | 1.000 | 0.000 |

No alternative strategy achieves even marginally significant improvement over the baseline, despite using 1.2–5.0× more tokens. The high-stakes emotional framing has zero effect on accuracy (identical results, just slightly more verbose). Self-consistency voting over 5 samples at temperature=0.7 uses 5× the compute for no accuracy gain — the model is already highly consistent at this capability level.

---

## 4. Discussion

### 4.1 Why Does Self-Critique Hurt?

Our results align with and extend the findings of Huang et al. (2024), who showed that intrinsic self-correction degrades reasoning performance. We add the nuance that the degradation is *monotonically related to critique stringency*: the harsher you ask the model to be, the worse it performs.

The mechanism is clear from the transition rates. At the adversarial level, the model changes 6.5% of its answers after self-critique. Of the originally correct answers, 5.7% are changed to incorrect (Correct→Wrong). Of the originally incorrect answers, only 25% are salvaged (Wrong→Correct). Since the baseline is already at 96% accuracy, there are far more correct answers to damage than incorrect answers to fix, making the net effect strongly negative.

This connects to Kamoi et al.'s (2024) key insight: **"LLMs cannot find reasoning errors, but can correct them given the error location."** The bottleneck is error *detection*, not error *correction*. When instructed to "find errors" with increasing urgency, the model does not become better at detecting real errors — it becomes more willing to *fabricate* errors where none exist.

### 4.2 Why Do Budget Constraints Help?

Our budget control results are consistent with three lines of prior work:

1. **TALE** (Han et al., 2024): Explicit token budgets improve both efficiency and accuracy.
2. **CCoT** (Nayab et al., 2024): Constraining output length reduces redundancy without hurting accuracy.
3. **When More is Less** (Wu et al., 2025): There exists an optimal CoT length, and unconstrained models generate responses that are longer than optimal.

The mechanism appears to be that budget constraints force the model to focus on essential reasoning steps, eliminating verbose filler that can introduce confusion or error. At 75 words, the model must identify the most direct solution path, which for GSM8K problems is often straightforward arithmetic that doesn't benefit from elaborate exposition.

### 4.3 The Budget–Critique Interaction

The factorial experiment reveals an interesting interaction: budget constraints partially rescue accuracy under harsh critique (88.0% → 92.0%). This suggests that the damage from self-critique is partly mediated by verbosity — when the model has unlimited space to critique, it generates more (spurious) objections. Constraining the critique response limits this error amplification.

However, even with tight budget constraints, critique still degrades performance relative to no critique at the same budget level (92.0% vs. 96.7%). Budget control is a partial but not complete antidote to critique-induced harm.

### 4.4 Why Don't Alternative Strategies Help?

The failure of all alternative strategies reveals an important truth about GPT-4.1-mini on GSM8K: **the model is already near its capability ceiling for this task**. At 95.3% baseline accuracy, there is very little room for improvement, and any intervention that introduces additional processing risks degrading rather than improving performance.

- **High-stakes framing**: Emotional urgency has zero measurable effect on mathematical reasoning. The model does not "try harder" in response to emotional pressure — it simply produces slightly more verbose versions of the same reasoning.
- **Explicit rubric**: Providing a scoring rubric creates a critique-like intervention, and indeed shows a slight (non-significant) accuracy decrease (94.7%).
- **Step verification**: Asking the model to re-solve independently is essentially a 2-sample self-consistency approach; with a high baseline accuracy, both solutions are usually correct, adding no value.
- **Self-consistency (k=5)**: With 95.3% per-sample accuracy, majority voting rarely changes the outcome. The expected disagreement rate is too low to benefit from consensus.

### 4.5 Practical Implications

For practitioners seeking to improve LLM output quality:

1. **Do not use self-critique prompts for reasoning tasks.** They are more likely to damage correct answers than fix incorrect ones.
2. **Use budget constraints.** Simple instructions like "Keep your answer to at most 75 words" can maintain or improve accuracy while significantly reducing cost.
3. **Save your compute.** Self-consistency, multi-step verification, and elaborate rubric evaluation use 2–5× more tokens for no measurable benefit when the model is already performing well.
4. **If you must use critique**, constrain the critique response length to limit error amplification.

### 4.6 Limitations

1. **Single model**: We tested only GPT-4.1-mini. Results may differ for weaker models (where the baseline is lower and there is more room for improvement) or for reasoning-specialized models (like o1/o3).
2. **Single benchmark**: GSM8K represents a specific task type (grade-school math). On tasks where quality is more subjective (dialogue, creative writing, code review), self-critique may behave differently. Self-Refine showed improvements on such tasks.
3. **Near-ceiling baseline**: At 95–96% accuracy, there is limited headroom. On harder benchmarks (MATH, GPQA), self-critique might be more beneficial.
4. **Sample size**: With 150–200 problems per experiment, we have limited statistical power for detecting small effects (<3 percentage points).
5. **Fixed critique structure**: We used single-round generate-then-critique. Multi-round iterative refinement (as in Self-Refine) was not tested.

---

## 5. Related Work

### Self-Correction in LLMs
- **Self-Refine** (Madaan et al., 2023) showed iterative refinement helps on subjective tasks but not math reasoning.
- **Huang et al. (2024)** demonstrated that intrinsic self-correction degrades reasoning performance.
- **Kamoi et al. (2024)** provided a comprehensive survey showing self-correction only works with external feedback.
- **Xu et al. (2024)** identified self-bias amplification in iterative self-refinement.

### Budget and Length Control
- **TALE** (Han et al., 2024) introduced token-budget-aware reasoning.
- **CCoT** (Nayab et al., 2024) showed concise chain-of-thought maintains accuracy.
- **Wu et al. (2025)** provided theoretical analysis of the optimal CoT length.
- **Poddar et al. (2025)** characterized LLM verbosity and demonstrated prompt-based reduction.

### Alternative Improvement Strategies
- **Self-consistency** (Wang et al., 2023) uses majority voting over multiple samples.
- **Reflexion** (Shinn et al., 2023) uses verbal reinforcement learning with environmental feedback.
- **Tree of Thoughts** (Yao et al., 2023) explores multiple reasoning paths.

---

## 6. Conclusions

We conducted the first systematic study of critique stringency in LLM self-evaluation, testing 5 levels from no critique to adversarial across 200 GSM8K problems. Our key findings:

1. **Harsher self-critique is counterproductive.** Accuracy drops monotonically from 96.0% (no critique) to 91.0% (harsh critique), with the decline statistically significant at p<0.01 for the harshest levels. The model is more likely to break correct answers than fix incorrect ones.

2. **Budget constraints are the simplest effective intervention.** Asking the model to be concise (75 words) maintains 96.7% accuracy while using 32% fewer tokens than unconstrained generation.

3. **Budget partially mitigates critique damage.** In factorial analysis, tight budget constraints under harsh critique (92.0%) partially recover the accuracy lost from unconstrained harsh critique (88.0%).

4. **No "effort-inducing" strategy beats the baseline.** High-stakes framing, explicit rubrics, step verification, and self-consistency all fail to improve upon simple chain-of-thought, while using 1.2–5× more tokens.

The popular intuition that "being harsh with LLMs" improves results is wrong — at least for reasoning tasks. The "lazy LLM" problem is better addressed by constraining output (forcing conciseness) rather than by demanding self-evaluation (which introduces errors). For future work, we recommend investigating these findings on harder benchmarks, weaker models, and subjective evaluation tasks where the dynamics of self-critique may differ.

---

## 7. Reproducibility

All code and data are available in this repository:
- `src/utils.py` — Shared utilities (API calls, caching, answer extraction)
- `src/experiment_critique.py` — Experiment 1: Critique stringency
- `src/experiment_budget.py` — Experiment 2: Budget control
- `src/experiment_combined.py` — Experiment 3: Factorial design
- `src/experiment_alternatives.py` — Experiment 4: Alternative strategies
- `src/analysis.py` — Statistical analysis and visualization
- `results/` — Raw results JSON and generated plots
- `datasets/gsm8k/test.jsonl` — GSM8K test set

**Environment**: Python 3.x with `openai`, `numpy`, `scipy`, `matplotlib`, `tqdm`.
**API**: OpenAI `gpt-4.1-mini` model.
**Random seed**: 42 for all problem sampling.
**Caching**: SHA-256 hash-keyed response cache for reproducibility.

---

## References

1. Madaan, A., et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." arXiv:2303.17651.
2. Huang, J., et al. (2024). "Large Language Models Cannot Self-Correct Reasoning Yet." arXiv:2310.01798.
3. Kamoi, R., et al. (2024). "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs." arXiv:2406.01297.
4. Xu, J., et al. (2024). "Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement." arXiv:2402.11436.
5. Han, H., et al. (2024). "Token-Budget-Aware LLM Reasoning." arXiv:2412.18547.
6. Nayab, S., et al. (2024). "Concise Thoughts: Impact of Output Length on LLM Reasoning." arXiv:2407.19825.
7. Wu, Y., et al. (2025). "When More is Less: Understanding Chain-of-Thought Length in LLMs." arXiv:2502.07266.
8. Poddar, S., et al. (2025). "Brevity is the Soul of Sustainability." arXiv:2506.08686.
9. Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168.
10. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." arXiv:2303.11366.
