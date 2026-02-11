# Resources Catalog: Fixing Lazy LLMs

Comprehensive inventory of all resources gathered for the "Fixing Lazy LLMs" research project.

---

## Papers (27 total)

All papers stored in `papers/` with PDF chunks for deep reading in `papers/pages/`.

### Tier 1: Deeply Read (Full Structured Notes Extracted)

| # | Title | Authors | Year | arXiv ID | File | Key Contribution |
|---|-------|---------|------|----------|------|-----------------|
| 1 | Self-Refine: Iterative Refinement with Self-Feedback | Madaan et al. | 2023 | 2303.17651 | `2303.17651_self_refine.pdf` | Foundational iterative self-critique framework; ~20% avg improvement across 7 tasks |
| 2 | Large Language Models Cannot Self-Correct Reasoning Yet | Huang et al. | 2024 | 2310.01798 | `2310.01798_cannot_self_correct.pdf` | Proves intrinsic self-correction degrades performance; external feedback required |
| 3 | When Can LLMs Actually Correct Their Own Mistakes? | Kamoi et al. | 2024 | 2406.01297 | `2406.01297_when_can_llms_correct.pdf` | Critical survey; taxonomy of self-correction; identifies methodological flaws in prior work |
| 4 | Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement | Xu et al. | 2024 | 2402.11436 | `2402.11436_pride_prejudice.pdf` | Self-bias amplifies during iterative refinement; external feedback reduces bias |
| 5 | Token-Budget-Aware LLM Reasoning (TALE) | Han et al. | 2024 | 2412.18547 | `2412.18547_token_budget_aware.pdf` | Token budget control improves both efficiency and accuracy; RLHF drives verbosity |
| 6 | Concise Chain-of-Thought (CCoT) | Nayab et al. | 2024 | 2407.19825 | `2407.19825_concise_thoughts.pdf` | Length constraints don't hurt accuracy; often improve it; 12-25% redundancy reduction |
| 7 | When More is Less: Understanding Chain-of-Thought Length | Wu et al. | 2025 | 2502.07266 | `2502.07266_when_more_is_less.pdf` | Inverted U-curve for CoT length vs accuracy; optimal length formula derived |
| 8 | Brevity is the Soul of Sustainability | Poddar et al. | 2025 | 2506.08686 | `2501.17946_brevity_sustainability.pdf` | LLM responses 3-20x too long; prompt strategies achieve 25-60% energy savings |

### Tier 2: Downloaded (Titles and Abstracts Reviewed)

| # | Title | arXiv ID | File | Topic |
|---|-------|----------|------|-------|
| 9 | Self-Critiquing Models for Assisting Human Evaluators | 2206.05802 | `2206.05802_self_critiquing.pdf` | Training models for critique |
| 10 | Reflexion: Language Agents with Verbal Reinforcement Learning | 2303.11366 | `2303.11366_reflexion.pdf` | Verbal self-reflection in episodic memory |
| 11 | Tree of Thoughts: Deliberate Problem Solving with LLMs | 2305.10601 | `2305.10601_tree_of_thoughts.pdf` | Structured deliberation via tree search |
| 12 | Self-Evaluation Guided Beam Search | 2305.00633 | `2305.00633_self_eval_beam_search.pdf` | Self-evaluation for decoding guidance |
| 13 | GPT-4 Doesn't Know It's Wrong | 2310.12397 | `2310.12397_gpt4_doesnt_know.pdf` | LLM inability to detect own errors |
| 14 | Can Large Language Models Provide Useful Feedback on Research Papers? | 2310.08118 | `2310.08118_can_llms_self_critique.pdf` | LLM self-critique capability assessment |
| 15 | Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives | 2401.02009 | `2401.02009_self_contrast.pdf` | Multiple perspectives for self-evaluation |
| 16 | Prompt Chaining for Stepwise Reasoning | 2406.00507 | `2406.00507_prompt_chaining_stepwise.pdf` | Decomposed reasoning chains |
| 17 | TICK: Checklists for LLM Evaluation | 2407.12753 | `2407.12753_tick_checklists.pdf` | Structured evaluation checklists |
| 18 | Internal Consistency and Self-Feedback in LLMs | 2407.14507 | `2407.14507_internal_consistency.pdf` | Consistency-based self-evaluation |
| 19 | Critic-CoT: Boosting Critical Thinking of LLMs | 2408.16326 | `2408.16326_critic_cot.pdf` | Chain-of-thought critique reasoning |
| 20 | MagiCore: Multi-Agent Group Interaction for Code Review | 2409.12147 | `2409.12147_magicore.pdf` | Multi-agent critique for code |
| 21 | Score-Based Self-Correction with RL | 2409.12917 | `2409.12917_score_self_correct_rl.pdf` | RL for self-correction training |
| 22 | DeCRIM: Detecting and Correcting Reasoning Inconsistencies in LLMs | 2410.06381 | `2410.06381_decrim.pdf` | Inconsistency detection and correction |
| 23 | s1: Simple Test-Time Scaling | 2501.19393 | `2501.19393_s1_simple_test_time.pdf` | Budget-controlled test-time compute |
| 24 | L1: Controlling LLM Reasoning Length | 2503.04697 | `2503.04697_l1_controlling_length.pdf` | Explicit length control for reasoning |
| 25 | Stop Overthinking: The Art of Efficient Reasoning | 2503.16419 | `2503.16419_stop_overthinking.pdf` | Reducing excessive reasoning |
| 26 | Deep Critic: Training Critique Models | 2505.15475 | `2505.15475_deep_critic.pdf` | Dedicated critique model training |
| 27 | Critique with GRPO | 2505.15875 | `2505.15875_critique_grpo.pdf` | Group relative policy optimization for critique |

---

## Datasets

All stored in `datasets/`. See `datasets/README.md` for detailed descriptions.

| Dataset | Location | Status | Size | Used By |
|---------|----------|--------|------|---------|
| **GSM8K** | `datasets/gsm8k/` | Complete | 4.7MB (7,473 train + 1,319 test) | Self-Refine, Cannot Self-Correct, TALE, CCoT, When Can Correct, When More is Less |
| **SVAMP** | `datasets/svamp/` | Complete | 333KB (1,000 problems) | CCoT |
| **CommonGen** | `datasets/commongen/` | Partial | 1.7MB (13,200 of ~67K rows) | Self-Refine, Cannot Self-Correct, Pride & Prejudice |
| **MATH** | `datasets/math/` | Info only | See DATASET_INFO.json | Pride & Prejudice, When More is Less |

### Datasets Referenced But Not Downloaded

| Dataset | Source | Used By | Notes |
|---------|--------|---------|-------|
| ASDIV | https://github.com/chaosking121/ASDiv | CCoT | Math word problems |
| CommonSenseQA | https://www.tau-nlp.sites.tau.ac.il/commonsenseqa | Cannot Self-Correct, When Can Correct | Commonsense QA |
| HotpotQA | https://hotpotqa.github.io/ | Cannot Self-Correct, When Can Correct | Multi-hop QA |
| Flores-200 | https://github.com/facebookresearch/flores | Pride & Prejudice | Machine translation |
| GPQA | https://github.com/idavidrein/gpqa | When More is Less | Graduate-level QA |
| MMLU STEM | https://github.com/hendrycks/test | When More is Less | Multi-task evaluation |
| FED | Mehri & Eskenazi 2020 | Self-Refine | Dialogue evaluation |
| PIE | Madaan et al. 2023 | Self-Refine | Code optimization |
| MathBench | - | TALE | Math benchmarking |
| LeetCode-2K | - | When More is Less | Programming problems |

---

## Code Repositories

All stored in `code/`. See `code/README.md` for descriptions.

| Repository | Paper | URL | Status | Key Files |
|-----------|-------|-----|--------|-----------|
| **self-refine** | Self-Refine (Madaan et al., 2023) | https://github.com/madaan/self-refine | Cloned | `src/` - task prompts; `data/` - datasets; `colabs/` |
| **TALE** | Token-Budget-Aware (Han et al., 2024) | https://github.com/GeniusHTX/TALE | Cloned | `TALE-EP.py`, `TALE-PT.py`, `search_budget.py` |
| **reflexion** | Reflexion (Shinn et al., 2023) | https://github.com/noahshinn/reflexion | Cloned | `programming_runs/`, `hotpotqa_runs/` |

### Additional Relevant Repositories (Not Cloned)

| Repository | Paper | URL | Notes |
|-----------|-------|-----|-------|
| selfrefine.info | Self-Refine | https://selfrefine.info/ | Project website with prompts and data |
| grade-school-math | GSM8K dataset | https://github.com/openai/grade-school-math | Dataset source |
| SVAMP | SVAMP dataset | https://github.com/arkilpatel/SVAMP | Dataset source |

---

## Paper Search Results

All search results stored in `paper_search_results/` as JSONL files.

| Search Query | File | Papers Found |
|-------------|------|-------------|
| lazy LLM response quality effort prompting critic self-critique | `lazy_LLM_response_quality_effort_prompting_critic_self-critique_20260210_201824.jsonl` | Multiple |
| LLM output quality improvement response budget control token length | `LLM_output_quality_improvement_response_budget_control_token_length_20260210_202011.jsonl` | Multiple |
| LLM self-refinement critique feedback evaluation generation quality | `LLM_self-refinement_critique_feedback_evaluation_generation_quality_20260210_202408.jsonl` | Multiple |
| prompt engineering techniques improve LLM reasoning chain-of-thought | `prompt_engineering_techniques_improve_LLM_reasoning_chain-of-thought_20260210_202627.jsonl` | Multiple |

---

## Key Findings Summary

### What the Literature Tells Us About "Fixing Lazy LLMs"

**The Problem is Real:**
- LLMs produce responses 3-20x longer than needed (Poddar et al., 2025)
- Single-pass outputs systematically underperform what models can achieve (Madaan et al., 2023)
- RLHF training incentivizes verbosity (Han et al., 2024)
- Accuracy follows an inverted U-curve with response length (Wu et al., 2025)

**What Doesn't Work:**
- Simple self-critique prompting for reasoning tasks (Huang et al., 2024)
- Iterative self-refinement without external signals (amplifies self-bias; Xu et al., 2024)
- Fine-tuning for brevity (actually increases length; Poddar et al., 2025)
- Generic feedback ("improve this") vs. specific feedback (Madaan et al., 2023)

**What Works:**
- Structured, specific, multi-aspect feedback with external verification (Madaan et al., 2023)
- Token budget control (Han et al., 2024; Nayab et al., 2024)
- Length-calibrated reasoning (Wu et al., 2025)
- External verification signals: code execution, search, constraint checking (Kamoi et al., 2024)
- Decomposition into independently verifiable sub-components (Kamoi et al., 2024)
- RL-based calibration naturally converges to optimal lengths (Wu et al., 2025)
- Cross-model critique (different model provides feedback; avoids self-bias)

**Key Open Questions:**
1. How does critique "harshness" interact with self-bias?
2. Can budget control and structured critique be combined?
3. What is the minimum training data needed for effective critique models?
4. Does task-adaptive budget allocation outperform fixed budgets?

---

## Project Structure

```
fix-lazy-llms-f21e-claude/
├── literature_review.md          # Comprehensive literature synthesis
├── resources.md                  # This file - resource catalog
├── pyproject.toml                # Project configuration
├── .resource_finder_complete     # Completion marker
├── papers/                       # 27 downloaded PDFs
│   ├── pages/                    # PDF chunks for deep reading
│   └── *.pdf                     # Full papers
├── paper_search_results/         # Search result JSONL files
├── datasets/                     # Downloaded datasets
│   ├── gsm8k/                    # Grade School Math 8K
│   ├── svamp/                    # SVAMP math problems
│   ├── commongen/                # CommonGen (partial)
│   ├── math/                     # MATH dataset info
│   ├── README.md
│   └── .gitignore
├── code/                         # Cloned repositories
│   ├── self-refine/              # Madaan et al. 2023
│   ├── TALE/                     # Han et al. 2024
│   ├── reflexion/                # Shinn et al. 2023
│   └── README.md
└── .venv/                        # Python virtual environment
```
