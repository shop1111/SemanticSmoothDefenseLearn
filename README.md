# SemanticSmooth Defense Learning Project

This project is a small, Kaggle-friendly reproduction scaffold for studying how
perplexity filtering and smoothing-based defenses behave against two different
jailbreak attack styles:

- **GCG-style attacks**: token-level adversarial suffixes that often look
  unnatural and have high perplexity.
- **AutoDAN-style attacks**: natural-language jailbreak prompts that are more
  readable, semantically meaningful, and often lower in perplexity.

The default model is `Qwen/Qwen2.5-1.5B-Instruct`.

## What This Project Tests

The experiment asks three practical questions:

1. Can a simple **PPL filter** catch adversarial prompts with unusually high
   perplexity?
2. Does that PPL filter work better on GCG-style prompts than AutoDAN-style
   prompts?
3. Can a lightweight **SemanticSmooth-lite** defense reduce attack success by
   perturbing prompts and voting over model responses?

PPL filtering is used as a baseline defense. SemanticSmooth-lite is the main
defense reproduction idea.

## Project Layout

```text
src/
  ppl_smoothing_defense_kaggle.py
reports/
  Kaggle_Qwen25_15B_PPL_Smoothing运行说明.md
results/
  qwen25_3b_hga_50.jsonl
examples/
  README.md
requirements.txt
```

`results/qwen25_3b_hga_50.jsonl` is kept as an AutoDAN result file for defense
evaluation input.

## Kaggle Setup

```bash
pip install -r requirements.txt
```

Check GPU availability:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Smoke Test

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/qwen25_3b_hga_50.jsonl \
  --limit 2 \
  --smooth-copies 3 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_defense_smoke.jsonl \
  --summary-output results/qwen25_15b_defense_smoke_summary.json
```

When no `--gcg-results` file is provided, the script adds a small built-in
GCG-like smoke set. Use real GCG results for formal reporting.

## AutoDAN Defense Run

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/qwen25_3b_hga_50.jsonl \
  --limit 50 \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_autodan_defense.jsonl \
  --summary-output results/qwen25_15b_autodan_defense_summary.json
```

## GCG + AutoDAN Comparison

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/qwen25_3b_hga_50.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl \
  --limit 50 \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_gcg_autodan_defense.jsonl \
  --summary-output results/qwen25_15b_gcg_autodan_defense_summary.json
```

## Main Output Fields

- `attack_type`: `gcg` or `autodan`
- `prompt_ppl`: perplexity of the attack prompt
- `ppl_blocked`: whether the PPL filter blocks the prompt
- `raw_success`: keyword-ASR success without defense
- `ppl_defended_success`: success after PPL filtering
- `smooth_success`: success after SemanticSmooth-lite voting
- `model_calls`: generation calls made by the defense script
- `elapsed_seconds`: per-sample runtime

## Recommended Report Table

| Attack | Defense | Avg PPL | Block Rate | ASR Before | ASR After | Avg Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCG | PPL Filter |  |  |  |  |  |
| GCG | SemanticSmooth-lite |  |  |  |  |  |
| AutoDAN | PPL Filter |  |  |  |  |  |
| AutoDAN | SemanticSmooth-lite |  |  |  |  |  |

## Notes

This is a course reproduction scaffold, not a full reproduction of the original
SemanticSmooth training pipeline. It focuses on the experimental idea: compare a
simple PPL baseline with smoothing-style prompt perturbation, then analyze why
semantic attacks are harder to catch than token-level suffix attacks.
