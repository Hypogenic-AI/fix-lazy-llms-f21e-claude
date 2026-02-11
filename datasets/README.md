# Datasets

Datasets gathered for the "Fixing Lazy LLMs" research project.

## Available Datasets

### GSM8K (Grade School Math 8K)
- **Source**: OpenAI - https://github.com/openai/grade-school-math
- **Files**: train.jsonl (7,473 problems), test.jsonl (1,319 problems)
- **Used in**: Self-Refine, Cannot Self-Correct, TALE, CCoT, Pride & Prejudice, When Can LLMs Correct
- **Description**: Grade school math word problems requiring multi-step reasoning

### SVAMP
- **Source**: https://github.com/arkilpatel/SVAMP
- **Files**: SVAMP.json
- **Used in**: CCoT (Concise Thoughts)
- **Description**: Simple Variations on Arithmetic Math word Problems

### CommonGen
- **Source**: Allen AI - https://inklab.usc.edu/CommonGen/
- **Used in**: Self-Refine (CommonGen-Hard variant), Cannot Self-Correct
- **Description**: Constrained text generation with concept sets

### MATH
- **Source**: Hendrycks et al. - https://github.com/hendrycks/math
- **Used in**: Pride & Prejudice, When More is Less
- **Description**: Competition-level mathematics problems across 7 subjects

## Notes
- Large binary files are excluded via .gitignore
- Some datasets require Hugging Face access or manual download
