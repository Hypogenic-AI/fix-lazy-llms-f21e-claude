# Research Plan: Fixing Lazy LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly deployed for tasks requiring careful, thorough work (code review, editing, analysis), yet users consistently report that models "take the easy way out" — producing generic, shallow, or verbose-but-unhelpful responses. If we can systematically improve output quality through prompt-based interventions (no retraining required), this has immediate practical value for millions of users.

### Gap in Existing Work
The literature review reveals a clear gap: **No paper systematically studies how the "harshness" or stringency of self-critique prompts affects output quality.** Self-Refine showed that *specific* feedback beats *generic* feedback, but didn't vary critique severity. Huang et al. showed intrinsic self-correction fails for reasoning, but tested only generic "review your answer" prompts. The interaction between critique intensity, budget control, and output quality is unexplored.

Additionally, the user's hypothesis about "rudeness" leading to better results connects to an unstudied dimension: does the emotional framing/urgency of prompts affect effort level?

### Our Novel Contribution
We conduct the first systematic study of:
1. **Critique stringency spectrum**: Varying self-critique from lenient to harsh across multiple tasks
2. **Budget control interaction**: Testing whether token budgets combined with critique improve results
3. **Effort-inducing prompt strategies**: Comparing multiple approaches to "fix laziness" (critique harshness, urgency/stakes framing, explicit quality rubrics, iterative refinement)
4. **Real model behavior**: Using actual state-of-the-art LLM APIs, not simulations

### Experiment Justification
- **Experiment 1 (Critique Stringency)**: Directly tests the user's core hypothesis — does asking an LLM to be a "harsher critic" improve output quality? No prior work has systematically varied this.
- **Experiment 2 (Budget Control)**: Tests whether constraining response length forces the model to be more efficient/accurate, drawing on TALE and CCoT findings but with modern models.
- **Experiment 3 (Combined Approaches)**: Tests whether combining critique + budget outperforms either alone — an open question identified in the literature.
- **Experiment 4 (Alternative Effort-Inducing Strategies)**: Tests whether other approaches (urgency framing, explicit rubrics, iterative refinement) compare to or outperform the critique approach.

---

## Research Question
**Primary**: Does prompting LLMs to act as increasingly harsh critics improve their output quality on tasks where they typically "take the easy way out"?

**Secondary**:
1. Does controlling response budget interact with critique stringency?
2. How does critique-based improvement compare to other effort-inducing strategies?
3. Is there an optimal level of critique harshness, or does quality monotonically increase?

## Hypothesis Decomposition

**H1**: Increasing critique stringency in self-evaluation prompts will improve output quality up to a point, with diminishing or negative returns at extreme harshness (inverted U-curve hypothesis).

**H2**: Token budget control will independently improve output quality by forcing conciseness, consistent with TALE and CCoT findings.

**H3**: Combining budget control with structured critique will outperform either approach alone.

**H4**: Multiple alternative strategies (high-stakes framing, explicit rubrics, iterative refinement) will show varying effectiveness, with structured approaches outperforming vague urgency.

## Proposed Methodology

### Approach
We use GSM8K (math reasoning) as our primary benchmark because:
- It has clear ground truth (numerical answers), enabling unambiguous evaluation
- It's the most widely used benchmark in the self-correction literature
- Math reasoning is where "laziness" manifests as shortcuts and errors
- It allows us to measure both accuracy AND reasoning quality

We additionally test on a subset of open-ended generation tasks (CommonGen) to measure quality on creative/generative tasks where "laziness" manifests differently.

### Models
- **GPT-4.1** via OpenAI API (primary model)
- **GPT-4.1-mini** via OpenAI API (cost-effective secondary model)
- We use two models to test generalizability across capability levels

### Experimental Steps

#### Experiment 1: Critique Stringency Spectrum
Test 5 levels of critique harshness on GSM8K (200 problems):

| Level | Critique Prompt Style |
|-------|----------------------|
| L0 (Baseline) | Direct answer, no self-critique |
| L1 (Gentle) | "Please review your answer and check for errors" |
| L2 (Moderate) | "Carefully examine each step. Identify any logical errors, calculation mistakes, or unjustified leaps." |
| L3 (Harsh) | "You are an extremely strict math professor grading an exam. Find every flaw, no matter how small. Assume there ARE errors and find them." |
| L4 (Adversarial) | "Your job is to DESTROY this solution. Find fatal flaws. Be merciless. This answer is probably wrong — prove it." |

Each level: Generate answer → Apply self-critique at specified level → Produce final answer.

#### Experiment 2: Budget Control
Test 4 budget levels on the same 200 GSM8K problems:

| Budget | Prompt Addition |
|--------|----------------|
| No limit | Standard CoT |
| Generous | "Solve in at most 300 words" |
| Moderate | "Solve in at most 150 words" |
| Tight | "Solve in at most 75 words" |

#### Experiment 3: Critique × Budget Interaction
Factorial design: 3 critique levels (none, moderate, harsh) × 3 budget levels (no limit, moderate, tight) = 9 conditions on 200 GSM8K problems.

#### Experiment 4: Alternative Effort-Inducing Strategies
Test on 200 GSM8K problems:

| Strategy | Prompt Approach |
|----------|----------------|
| Baseline | Standard CoT |
| High Stakes | "This is an extremely important exam. Your career depends on getting this right. Take your time and be thorough." |
| Explicit Rubric | Provide a detailed scoring rubric and ask model to evaluate against it |
| Step-by-Step Verification | "After solving, verify each step by plugging values back in" |
| Best-of-N | Generate 5 answers, pick most common (self-consistency baseline) |

### Baselines
1. **Direct prompting** (zero-shot CoT): "Let's think step by step"
2. **Self-consistency** (majority vote over 5 samples at temp=0.7): Equivalent compute comparison
3. **Simple self-review**: "Review your answer and correct any mistakes"

### Evaluation Metrics

**GSM8K (Math)**:
- **Accuracy**: Exact match of final numerical answer (primary metric)
- **Response length**: Token count (to measure verbosity)
- **Reasoning quality**: Manual annotation of 50 samples for logical coherence (1-5 scale)

**Across all experiments**:
- Cost (API tokens consumed)
- Answer change rate (how often critique leads to changing the answer)
- Correct→Wrong rate (how often critique breaks a correct answer)
- Wrong→Correct rate (how often critique fixes an incorrect answer)

### Statistical Analysis Plan
- **Primary test**: McNemar's test for paired accuracy comparisons (binary correct/incorrect)
- **Multiple comparisons**: Bonferroni correction across pairwise tests
- **Effect sizes**: Cohen's h for proportion differences
- **Confidence intervals**: 95% Wilson score intervals for proportions
- **Significance level**: α = 0.05

## Expected Outcomes

**If H1 is supported**: We expect an inverted U-curve where moderate critique (L2-L3) outperforms both no critique (L0) and adversarial critique (L4). The adversarial level may cause the model to second-guess correct answers.

**If H2 is supported**: Moderate budget constraints will improve accuracy slightly while reducing token usage significantly.

**If H3 is supported**: The combination of moderate critique + moderate budget will be the best overall condition.

**Alternative outcome**: If critique uniformly hurts performance (consistent with Huang et al.), this still advances knowledge by testing across stringency levels rather than just on/off.

## Timeline and Milestones

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Planning | 15 min | planning.md |
| Environment setup | 10 min | Working environment |
| Implementation | 45 min | Experiment scripts |
| Experiment 1 (Critique) | 30 min | Results for 5 conditions |
| Experiment 2 (Budget) | 20 min | Results for 4 conditions |
| Experiment 3 (Combined) | 30 min | Results for 9 conditions |
| Experiment 4 (Alternatives) | 25 min | Results for 5 conditions |
| Analysis & Visualization | 30 min | Plots and statistics |
| Documentation | 20 min | REPORT.md, README.md |

## Potential Challenges

1. **API rate limits**: Mitigate with exponential backoff and parallel requests
2. **Cost**: ~200 problems × ~25 conditions × ~2 models = ~10,000 API calls. Use GPT-4.1-mini for most conditions.
3. **Critique not changing answers**: Track answer-change rate to understand intervention strength
4. **Model refusal at adversarial levels**: The model may refuse to "destroy" its own answer; handle gracefully

## Success Criteria

1. At least 3 experiments complete with statistically meaningful sample sizes
2. Clear comparison across conditions with statistical tests
3. Identification of whether critique stringency has a monotonic or non-monotonic relationship with accuracy
4. Practical recommendation for users who want to improve LLM output quality
