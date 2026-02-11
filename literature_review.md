# Literature Review: Fixing Lazy LLMs

## Research Question

**Hypothesis:** Large language models (LLMs) tend to prefer easy or less effortful responses because they lack subjective judgment of good or bad; prompting LLMs to act as harsher critics and controlling the response budget may improve their output quality.

**Core Questions:**
1. Do LLMs systematically produce suboptimal outputs when more capable responses are achievable?
2. Can self-critique and iterative refinement improve output quality?
3. Does controlling response length/budget affect quality?
4. What are the limitations of intrinsic self-correction, and what alternatives exist?

---

## 1. The "Lazy LLM" Phenomenon: Evidence and Characterization

### 1.1 LLMs Produce Verbose but Low-Quality Initial Outputs

Multiple papers provide direct evidence that LLMs default to suboptimal generation strategies:

**Self-Refine** (Madaan et al., 2023; arXiv:2303.17651) demonstrated that LLMs consistently produce outputs that fall short of their actual capability. Across 7 tasks, Self-Refine improved outputs by an average of ~20% absolute through iterative self-feedback, proving that single-pass generation systematically underperforms what the model can achieve. Concrete examples include: dialogue responses that are "generic and uninformative" on first pass but become detailed and engaging after refinement; code that uses brute-force algorithms initially but transforms into efficient dynamic programming solutions after feedback.

**Brevity is the Soul of Sustainability** (Poddar et al., 2025; arXiv:2506.08686) benchmarked 12 LLMs across 5 datasets and found that LLM responses are **3-20x longer than necessary** for factual questions. Phi-3 models produced responses up to 418x longer than target answers. Only 42% of response content constituted the "minimal answer" -- the rest was additional information (21%), explanations (11.5%), irrelevant content (~18%), conversational filler (5.2%), and redundancy. Simple prompt strategies ("Answer briefly") achieved 25-60% energy savings while *improving* quality as measured by ROUGE-L F1.

**When More is Less** (Wu et al., 2025; arXiv:2502.07266) provided the strongest theoretical evidence: task accuracy follows an **inverted U-shaped curve** with Chain-of-Thought (CoT) length. For a 72B parameter model, the gap between optimal-length CoT accuracy and longest-CoT accuracy was **40 percentage points**. Before RL training, models generate CoTs that are longer than optimal; RL training causes CoT length to *decrease* as accuracy improves.

### 1.2 Training Incentives Drive Verbosity

Several papers identify training mechanisms that produce verbose behavior:

- **RLHF bias toward verbosity**: The Token-Budget-Aware (TALE) paper (Han et al., 2024; arXiv:2412.18547) explicitly notes that RLHF training biases models toward longer outputs because human evaluators tend to prefer longer responses during reward model training, creating a systematic incentive for verbosity.
- **Fine-tuning cannot easily override verbosity**: Poddar et al. (2025) found that LoRA fine-tuning on target-length answers actually *increased* response length by 24-26%, suggesting that pre-training patterns for verbosity are deeply ingrained and resist small-scale adaptation.
- **Newer models are more verbose**: The Brevity paper found that newer models (Llama-3.1, GPT-4o) generate longer responses than their predecessors, suggesting that recent training strategies amplify verbosity.

### 1.3 The Simplicity Bias

**When More is Less** established a key finding: more capable models achieve peak performance with *shorter* reasoning chains. For Qwen2.5 on MATH Level 5, the optimal CoT length dropped from 14 steps (1.5B model) to 4 steps (72B model). This "simplicity bias" means that improving model capability should be paired with mechanisms that encourage more concise reasoning, not more tokens.

---

## 2. Self-Critique and Iterative Refinement

### 2.1 Self-Refine: The Positive Case

**Self-Refine** (Madaan et al., 2023) is the foundational framework for iterative self-improvement. The approach uses a single LLM as generator, feedback provider, and refiner through three phases:

1. **Generate**: Produce initial output y₀
2. **Feedback**: Generate structured, multi-aspect, actionable feedback on y₀
3. **Refine**: Generate improved output y₁ using the feedback; iterate

Key results across 7 tasks (sentiment reversal, dialogue, code optimization, code readability, math reasoning, acronym generation, constrained generation):
- GPT-4 + Self-Refine showed the largest gains: +49.2% on dialogue, +32.4% on sentiment reversal, +30.0% on constrained generation
- Self-Refine consistently outperformed sampling k=4 independent outputs, proving that iterative feedback-guided refinement is more effective than generating more candidates
- **Critical ablation**: Specific feedback (e.g., "avoid repeated calculations in the for loop") dramatically outperformed generic feedback ("improve the efficiency"), which outperformed no feedback at all
- Most gains came from the first iteration; improvements diminished but continued through 3-4 iterations

**Limitations**: Self-Refine showed near-zero improvement on math reasoning (ChatGPT gave "everything looks good" feedback 94% of the time). Weaker models (Vicuna-13B) failed to generate structured feedback consistently.

**Code and data**: https://selfrefine.info/, https://github.com/madaan/self-refine

### 2.2 Cannot Self-Correct: The Negative Case

**Large Language Models Cannot Self-Correct Reasoning Yet** (Huang et al., 2024; arXiv:2310.01798) directly challenged the premise of self-correction:

- **Central finding**: LLMs cannot self-correct reasoning without external feedback. "Intrinsic self-correction" (prompting the same model to review its own answer) consistently **degrades** performance.
- On GSM8K with GPT-3.5-Turbo, self-correction changed 12.3% of correct answers to incorrect but only changed 1.7% of incorrect answers to correct (net negative).
- Multi-agent debate did not outperform self-consistency (majority voting over multiple samples).
- **External feedback is necessary**: When oracle feedback (indicating an answer is wrong) was provided, self-correction became effective. The problem is not the correction mechanism but the error detection mechanism.

**Datasets**: GSM8K, CommonSenseQA, HotpotQA, CommonGen-Hard

### 2.3 When Can LLMs Actually Correct Their Own Mistakes?

**A Critical Survey of Self-Correction of LLMs** (Kamoi et al., 2024; arXiv:2406.01297) provided the most comprehensive analysis, organizing self-correction along three dimensions (feedback source, architecture, experimental fairness):

**When self-correction works:**
- Tasks with **decomposable responses** where sub-answers can be verified independently (e.g., checking if a named politician was actually born in NY)
- **Reliable external feedback** from code interpreters, search engines, proof assistants
- **Large-scale fine-tuning** (100K+ instances) for feedback generation
- Reinforcement learning approaches (e.g., OpenAI o1)

**When self-correction fails:**
- **Intrinsic self-correction on general tasks**: No prior work demonstrates success under fair experimental conditions
- Specific failures: arithmetic reasoning (GSM8K), closed-book QA (CSQA, HotpotQA), code generation, plan generation, graph coloring, logical reasoning

**Critical methodological finding**: Many "positive" results for self-correction used unfair experimental setups:
- Self-Refine used prompts that *intentionally sabotage* initial responses (dialogue prompt instructed "The response is not interesting" and "The response is not very engaging")
- RCI Prompting used ground-truth answers as stopping conditions
- Reflexion used exact-match with ground truth as feedback

**Key insight**: "LLMs cannot find reasoning errors, but can correct them given the error location" -- the bottleneck is error *detection*, not error *correction*.

### 2.4 Pride and Prejudice: Self-Bias in Refinement

**Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement** (Xu et al., 2024; arXiv:2402.11436) revealed a fundamental obstacle to self-improvement:

- LLMs systematically overestimate the quality of their own output ("self-bias")
- This bias **amplifies** during iterative self-refinement, creating a positive feedback loop
- Self-refinement improves surface fluency but does NOT improve task-level quality
- External feedback with accurate quality assessment dramatically reduces bias

**Datasets**: Flores-200 (translation), CommonGen Hard, MATH

---

## 3. Controlling Response Budget and Length

### 3.1 Token-Budget-Aware LLM Reasoning (TALE)

**TALE** (Han et al., 2024; arXiv:2412.18547) introduced a framework for compressing CoT reasoning via explicit token budgets:

- **TALE-EP** (Estimating-Prompting): Training-free approach that estimates task difficulty and sets an appropriate token budget
- **TALE-PT** (Post-Training): Fine-tunes models to generate efficient reasoning within budget constraints
- **Key finding**: On GSM8K, TALE-EP actually *outperformed* vanilla CoT while using fewer tokens, demonstrating that budget control can improve both efficiency and accuracy
- RLHF-trained models systematically produce longer outputs than necessary; explicit budget control counteracts this tendency

**Code**: https://github.com/GeniusHTX/TALE
**Datasets**: GSM8K, GSM8K-Zero, MathBench

### 3.2 Concise Chain-of-Thought (CCoT)

**CCoT** (Nayab et al., 2024; arXiv:2407.19825) demonstrated that constraining output length does NOT hurt accuracy -- and often improves it:

- Simple prompting strategy: append "and limit the answer length to [N] words" to the prompt
- Achieved 12-25% reduction in redundancy with maintained or improved accuracy
- Verbosity is not a proxy for reasoning quality; shorter, more focused responses often capture the essential reasoning without filler

**Datasets**: GSM8K, SVAMP, ASDIV

### 3.3 The Optimal Length Exists and Is Computable

**When More is Less** (Wu et al., 2025) provided the theoretical foundation:

- Derived a closed-form expression for optimal CoT length: N*(M,T) = TZ / M(Z+1), where Z involves the Lambert W function, M is model capability, and T is task difficulty
- Optimal length increases with task difficulty but decreases with model capability
- **Practical implication**: Length-Filtered Vote (grouping CoT solutions by length, computing entropy, voting over lowest-entropy groups) consistently outperformed vanilla majority voting on GPQA
- Training with optimal-length CoT data: a smaller model (6-layer GPT-2) trained on optimal-length data outperformed a larger model (9-layer GPT-2) trained on mixed-length data

---

## 4. External Feedback and Verification

### 4.1 The Case for External Feedback

The literature converges on a clear conclusion: **external feedback is necessary for reliable self-correction**. The error detection bottleneck cannot be overcome by prompting alone for general tasks.

Effective external feedback sources identified across papers:
- **Code interpreters/compilers**: Deterministic pass/fail signals (Self-Debug, CRITIC, Reflexion for code)
- **Search engines**: Factual verification via retrieved documents
- **Proof assistants**: Formal correctness verification
- **Programmatic constraint checking**: For constrained generation tasks
- **Human feedback**: Gold standard but expensive
- **Stronger model feedback**: "Mixed-refine" approach (Vicuna-13B initial + ChatGPT feedback = 24.18% → 40.5% on math)

### 4.2 Reflexion: Verbal Reinforcement Learning

**Reflexion** (Shinn et al., 2023; arXiv:2303.11366) introduced verbal self-reflection stored in episodic memory:
- The agent generates actions, receives environment feedback, produces verbal reflections, and uses those reflections in subsequent attempts
- Effective for sequential decision-making, code generation, and reasoning tasks
- **Important caveat** (per Kamoi et al.): Uses exact-match with ground truth as feedback in some experiments, making the experimental setup "unfair" for evaluating intrinsic self-correction

### 4.3 Structured Critique Frameworks

Several papers explore structured approaches to critique:
- **TICK: Checklists** (arXiv:2407.12753): Using structured checklists for systematic evaluation
- **Critic CoT** (arXiv:2408.16326): Chain-of-thought critique reasoning
- **Self-Contrast** (arXiv:2401.02009): Generating multiple diverse perspectives to create more reliable self-evaluation
- **Deep Critic** (arXiv:2505.15475): Training dedicated critique models
- **Critique with GRPO** (arXiv:2505.15875): Using group relative policy optimization for critique training

---

## 5. Synthesis: What Works for Fixing Lazy LLMs?

### 5.1 What Doesn't Work

1. **Simple self-critique prompting** ("review your answer and fix any errors"): Fails consistently on reasoning tasks under fair conditions. LLMs cannot reliably detect their own errors.

2. **Vague instructions to be more critical**: Generic feedback performs far worse than specific, actionable feedback (Self-Refine ablation).

3. **Assuming more tokens = more quality**: Both theoretical (When More is Less) and empirical (CCoT, Brevity) evidence shows that verbosity degrades quality past an optimal point.

4. **Fine-tuning for brevity**: Small-scale fine-tuning cannot override deep-seated verbosity patterns from pre-training (Brevity paper).

5. **Iterative self-refinement without external signals**: Self-bias amplifies over iterations, improving fluency while degrading task quality (Pride and Prejudice).

### 5.2 What Works

1. **Structured, specific feedback with external verification**: When feedback is actionable, specific, and grounded in external evidence (code execution, search results, constraint checking), self-correction is highly effective.

2. **Token budget control**: Explicitly setting response budgets (TALE, CCoT) can improve both efficiency and accuracy. The key insight is that forcing conciseness eliminates redundant tokens while preserving essential reasoning.

3. **Optimal-length calibration**: Training or prompting for task-appropriate reasoning length (not too short, not too long) outperforms both lazy and verbose strategies (When More is Less).

4. **Multi-aspect, specific critique prompts**: Self-Refine's success depended on feedback that was (a) actionable, (b) specific to exact locations/elements, and (c) multi-dimensional. This is much more effective than "be more critical."

5. **Decomposition for verification**: Breaking outputs into independently verifiable sub-components makes self-correction effective even without external tools (CoVe, FActScore approach).

6. **RL-based calibration**: Reinforcement learning naturally drives models toward optimal reasoning lengths and can recalibrate verbosity even from suboptimal starting points.

7. **Cross-model critique**: Using a different (potentially smaller, specialized) model for critique avoids the self-bias problem while maintaining the benefit of model-based feedback.

### 5.3 The Reconciliation

The apparent contradiction between Self-Refine (self-critique works) and Huang et al. (self-critique doesn't work) resolves when we consider task type and feedback structure:

- **Self-Refine works** on tasks where quality is multi-dimensional and the model can identify specific aspects to improve (dialogue, code readability, constrained generation). These are tasks where the model has latent knowledge about quality that isn't expressed in single-pass generation.

- **Self-correction fails** on tasks where error detection requires capabilities the model lacks (mathematical reasoning, factual verification). The model cannot identify what it doesn't know.

- **The critical variable is feedback quality**, not the act of self-reflection itself. External verification, structured evaluation criteria, and decomposition all improve feedback quality, which is the bottleneck.

---

## 6. Gaps and Opportunities

### 6.1 Under-Explored Areas

1. **Calibrated harshness in critique**: No paper systematically studies how the "harshness" or stringency of self-critique prompts affects output quality. The Self-Refine ablation (specific vs. generic feedback) is the closest, but direct manipulation of critique severity is unexplored.

2. **Task-adaptive budget control**: While TALE and CCoT demonstrate fixed budget benefits, dynamic per-instance budget allocation based on estimated difficulty is underexplored.

3. **The interaction between critique strength and self-bias**: If harsher critique prompts trigger stronger self-bias (defensive responses), there may be a non-monotonic relationship between critique intensity and improvement.

4. **Small training data regimes for critique models**: Most fine-tuning approaches require 100K+ instances. Few-shot or low-resource critique training is underexplored.

5. **Combining budget control with structured critique**: No paper integrates token budget constraints with iterative self-refinement.

### 6.2 Experimental Design Recommendations

Based on the methodological critiques in Kamoi et al. (2024):

1. **Always use strong initial prompts** -- never compare self-correction against deliberately weakened baselines
2. **Compare against self-consistency** (majority voting) at equivalent compute cost
3. **Report feedback quality metrics** (error detection accuracy), not just downstream task performance
4. **Use fair experimental frameworks**: same model, same information for both initial and corrected responses
5. **Test on multiple task types**: reasoning, generation, constrained, open-ended

---

## 7. Key Datasets Across Papers

| Dataset | Papers Using It | Task Type |
|---------|----------------|-----------|
| GSM8K | Self-Refine, Cannot Self-Correct, TALE, CCoT, When Can Correct, When More is Less | Math reasoning |
| MATH | Pride & Prejudice, When More is Less | Competition math |
| SVAMP | CCoT | Math word problems |
| ASDIV | CCoT | Math word problems |
| CommonSenseQA | Cannot Self-Correct, When Can Correct | Commonsense QA |
| HotpotQA | Cannot Self-Correct, When Can Correct | Multi-hop QA |
| CommonGen-Hard | Self-Refine, Cannot Self-Correct, Pride & Prejudice | Constrained generation |
| Flores-200 | Pride & Prejudice | Machine translation |
| GPQA | When More is Less | Graduate-level QA |
| MMLU STEM | When More is Less | Multi-task science |
| FED | Self-Refine | Dialogue evaluation |
| PIE | Self-Refine | Code optimization |

---

## 8. Key Code Repositories

| Repository | Paper | URL |
|-----------|-------|-----|
| self-refine | Self-Refine (Madaan et al., 2023) | https://github.com/madaan/self-refine |
| TALE | Token-Budget-Aware (Han et al., 2024) | https://github.com/GeniusHTX/TALE |
| reflexion | Reflexion (Shinn et al., 2023) | https://github.com/noahshinn/reflexion |

---

## 9. Paper Catalog

### Tier 1: Deeply Read (Full Structured Notes Available)

1. **Self-Refine** (Madaan et al., 2023) -- arXiv:2303.17651 -- Iterative refinement with self-feedback
2. **Cannot Self-Correct** (Huang et al., 2024) -- arXiv:2310.01798 -- LLMs cannot self-correct reasoning without external feedback
3. **When Can LLMs Correct** (Kamoi et al., 2024) -- arXiv:2406.01297 -- Critical survey of self-correction
4. **Pride and Prejudice** (Xu et al., 2024) -- arXiv:2402.11436 -- Self-bias amplification in self-refinement
5. **Token-Budget-Aware** (Han et al., 2024) -- arXiv:2412.18547 -- TALE framework for budget-controlled reasoning
6. **Concise Thoughts** (Nayab et al., 2024) -- arXiv:2407.19825 -- CCoT: constraining output length improves quality
7. **When More is Less** (Wu et al., 2025) -- arXiv:2502.07266 -- Inverted U-curve for CoT length vs. accuracy
8. **Brevity is the Soul of Sustainability** (Poddar et al., 2025) -- arXiv:2506.08686 -- LLM response length characterization and prompt-based reduction

### Tier 2: Downloaded, Titles and Abstracts Reviewed

9. **Self-Critiquing Models** -- arXiv:2206.05802
10. **Reflexion** (Shinn et al., 2023) -- arXiv:2303.11366
11. **Tree of Thoughts** -- arXiv:2305.10601
12. **Self-Eval Beam Search** -- arXiv:2305.00633
13. **GPT-4 Doesn't Know It's Wrong** -- arXiv:2310.12397
14. **Can LLMs Self-Critique** -- arXiv:2310.08118
15. **Self-Contrast** -- arXiv:2401.02009
16. **Prompt Chaining Stepwise** -- arXiv:2406.00507
17. **TICK Checklists** -- arXiv:2407.12753
18. **Internal Consistency** -- arXiv:2407.14507
19. **Critic CoT** -- arXiv:2408.16326
20. **MagiCore** -- arXiv:2409.12147
21. **Score Self-Correct RL** -- arXiv:2409.12917
22. **DeCRIM** -- arXiv:2410.06381
23. **S1 Simple Test-Time** -- arXiv:2501.19393
24. **L1 Controlling Length** -- arXiv:2503.04697
25. **Stop Overthinking** -- arXiv:2503.16419
26. **Deep Critic** -- arXiv:2505.15475
27. **Critique with GRPO** -- arXiv:2505.15875

---

## 10. Implications for Experimental Design

Based on this review, a research program on "fixing lazy LLMs" should:

1. **Test the "harsher critic" hypothesis carefully**: Vary critique stringency systematically while controlling for the Self-Refine finding that *specificity* matters more than *severity*.

2. **Combine budget control with critique**: Test whether setting explicit token budgets during self-refinement iterations produces better results than either approach alone.

3. **Use fair baselines**: Always compare against (a) direct prompting with strong prompts, (b) self-consistency at equivalent compute, (c) generate-and-rank approaches.

4. **Measure feedback quality directly**: Report error detection accuracy and feedback actionability, not just downstream task metrics.

5. **Test across the laziness spectrum**: Include tasks where models are too brief (lazy) and tasks where models are too verbose (overthinking) to understand whether the same interventions work for both.

6. **Consider the inverted-U**: Any intervention that increases output length must account for the theoretical result that there is an optimal length beyond which quality degrades.
