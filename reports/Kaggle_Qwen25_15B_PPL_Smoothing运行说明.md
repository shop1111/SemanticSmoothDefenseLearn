# Kaggle Qwen2.5-1.5B PPL Baseline + SemanticSmooth-lite 运行说明

## 用途

`src/ppl_smoothing_defense_kaggle.py` 用于比较 GCG 类 token-level suffix attack 和 AutoDAN 类 semantic jailbreak attack 在两种防御下的表现：

- PPL filter baseline：按 prompt perplexity 阈值拦截异常 prompt。
- SemanticSmooth-lite：对 prompt 做轻量语义扰动，多次生成后用拒答关键词投票。

默认模型统一为 `Qwen/Qwen2.5-1.5B-Instruct`。如果没有 GCG 结果文件，脚本会自动加入少量 GCG-like smoke 样例，正式报告应替换为真实 GCG 复现输出。

## Kaggle 准备

```bash
pip install transformers==4.48.3 accelerate sentencepiece protobuf tqdm bitsandbytes
```

验证 GPU：

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

该命令会读取 2 条 AutoDAN 结果，并自动加入 GCG-like smoke 样例。重点确认：

- `prompt_ppl` 是有效数值。
- `ppl_blocked`、`ppl_defended_success`、`smooth_success` 字段存在。
- 末尾 summary 中有 `autodan` 和 `gcg` 两组。

## AutoDAN 正式防御实验

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

## GCG + AutoDAN 对比实验

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

GCG 文件支持 JSONL 或 CSV。推荐字段：

- `goal`
- `adversarial_suffix` 或 `prompt`
- `response`
- `success`

如果只有完整攻击 prompt，提供 `prompt` 即可。

## PPL 阈值

默认使用脚本内置 benign prompts 的第 95 分位数作为阈值。若有自己的 benign/dev 集：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --benign-prompts data/benign_prompts.csv \
  --ppl-percentile 95 \
  ...
```

也可以手动指定：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --ppl-threshold 120 \
  ...
```

报告中需要记录 `ppl_threshold` 和 `ppl_threshold_source`。

## 输出解读

每条 JSONL 包含：

- `attack_type`：`gcg` 或 `autodan`
- `prompt_ppl`：攻击 prompt 的困惑度
- `ppl_blocked`：是否被 PPL filter 拦截
- `raw_success`：无防御时 keyword-ASR 是否成功
- `ppl_defended_success`：PPL filter 后是否仍成功
- `smooth_success`：SemanticSmooth-lite 投票后是否仍成功
- `model_calls`：本脚本对目标模型的生成调用次数
- `elapsed_seconds`：该样本耗时

Summary 中重点填入报告表格：

- `avg_ppl`
- `ppl_block_rate`
- `raw_asr`
- `ppl_defended_asr`
- `smooth_asr`
- `avg_model_calls`
