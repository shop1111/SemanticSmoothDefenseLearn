# SemanticSmooth Defense Learning Project

This project is a Kaggle-friendly reproduction scaffold for studying how
perplexity filtering and smoothing-based defenses behave against two jailbreak
attack styles:

- **GCG-style attacks**: token-level adversarial suffixes that often look
  unnatural and have high perplexity.
- **AutoDAN-style attacks**: natural-language jailbreak prompts that are more
  readable, semantically meaningful, and often lower in perplexity.

The default model is `Qwen/Qwen2.5-1.5B-Instruct`.

## Project Layout

```text
src/
  ppl_smoothing_defense_kaggle.py        # PPL filter + SemanticSmooth-lite evaluation
  prepare_defense_training_inputs.py     # Normalize GCG/AutoDAN artifacts
reports/
  Kaggle_Qwen25_15B_PPL_Smoothing运行说明.md
results/
  gcg_qwen25_15b.jsonl                   # Formal GCG input, 25 rows
  autodan/
    autodan_final_normalized.jsonl        # Formal AutoDAN input, 10 rows
  defense_training_inputs.jsonl           # Unified attack-positive inputs, 35 rows
  defense_training_inputs_summary.json
  defense_training_inputs_summary.csv
  archive/                                # Legacy/intermediate generation artifacts
examples/
  README.md
requirements.txt
```

The main Kaggle run should use:

- `results/gcg_qwen25_15b.jsonl`
- `results/autodan/autodan_final_normalized.jsonl`

`results/archive/qwen25_3b_hga_50_legacy_hga.jsonl` is kept only as a legacy
input reference. AutoDAN generation intermediates are under
`results/archive/autodan_generation/`.

## Setup

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
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl \
  --limit 2 \
  --smooth-copies 2 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_smoke_defense_run.jsonl \
  --summary-output results/qwen25_15b_smoke_defense_summary.json
```

## GCG + AutoDAN Defense Run

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl \
  --limit 25 \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_gcg_autodan_defense_run.jsonl \
  --summary-output results/qwen25_15b_gcg_autodan_defense_summary.json
```

`--limit 25` reads up to 25 rows from each attack file. With the current inputs,
the run evaluates 25 GCG rows and 10 AutoDAN rows.

## Preparing Inputs Again

If you have new external GCG or AutoDAN artifacts, normalize them with:

```bash
python src/prepare_defense_training_inputs.py \
  --gcg-artifact /path/to/GCG/Qwen2.5-1.5B-Instruct.json \
  --gcg-eval /path/to/eval_local_details.json \
  --autodan /path/to/autodan_final.jsonl
```

Default outputs:

- `results/gcg_qwen25_15b.jsonl`
- `results/autodan/autodan_final_normalized.jsonl`
- `results/defense_training_inputs.jsonl`
- `results/defense_training_inputs_summary.json`
- `results/defense_training_inputs_summary.csv`

The unified training file currently contains attack-positive examples only.
Add benign/refusal/ordinary prompts before training a binary defense classifier.

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
SemanticSmooth training pipeline. It focuses on comparing a simple PPL baseline
with smoothing-style prompt perturbation and explaining why semantic attacks are
harder to catch than token-level suffix attacks.
