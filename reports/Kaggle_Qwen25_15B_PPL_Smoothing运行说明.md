# Kaggle Qwen2.5-1.5B PPL Baseline + SemanticSmooth-lite 运行说明

## 1. 项目用途

`src/ppl_smoothing_defense_kaggle.py` 用于比较两类 jailbreak attack 在两种轻量防御下的表现：

- GCG 类 token-level suffix attack。
- AutoDAN 类 natural-language semantic jailbreak attack。

防御方法：

- PPL filter baseline：按 prompt perplexity 阈值拦截异常 prompt。
- SemanticSmooth-lite：对 prompt 做轻量语义扰动，多次生成后用拒答关键词投票。

默认模型是 `Qwen/Qwen2.5-1.5B-Instruct`。

## 2. 当前正式输入路径

Kaggle 正式运行只需要下面两个攻击输入文件：

```text
results/gcg_qwen25_15b.jsonl
results/autodan/autodan_final_normalized.jsonl
```

当前数据规模：

```text
GCG: 25 rows
AutoDAN: 10 rows
Unified attack-positive inputs: 35 rows
```

辅助文件：

```text
results/defense_training_inputs.jsonl
results/defense_training_inputs_summary.json
results/defense_training_inputs_summary.csv
```

归档文件：

```text
results/archive/qwen25_3b_hga_50_legacy_hga.jsonl
results/archive/autodan_generation/
```

归档文件用于追溯旧输入和中间生成结果，不参与正式 Kaggle 命令。

## 3. 在 Kaggle 创建 Notebook

1. 打开 Kaggle，创建一个新 Notebook。
2. 右侧 `Settings` 中打开 GPU。
3. 推荐 GPU：`GPU T4 x2` 或至少一张 T4。
4. 如果需要联网安装依赖和下载 Hugging Face 模型，打开 `Internet`。
5. 语言选择 Python。

## 4. 获取项目代码

方式 A：直接从 GitHub 克隆。

```bash
git clone https://github.com/shop1111/SemanticSmoothDefenseLearn.git
cd SemanticSmoothDefenseLearn
```

方式 B：上传 ZIP 到 Kaggle Dataset 后解压。

```bash
unzip /kaggle/input/YOUR_DATASET_NAME/SemanticSmoothDefenseLearn.zip -d /kaggle/working/
cd /kaggle/working/SemanticSmoothDefenseLearn
```

如果使用方式 A，建议先确认文件存在：

```bash
ls
ls results
ls results/autodan
```

## 5. 安装依赖

```bash
pip install -r requirements.txt
```

如果 Kaggle 上 `bitsandbytes` 安装或加载失败，可以先用不量化的默认命令运行；本项目默认没有开启 `--load-in-4bit`。

## 6. 验证 GPU

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

如果输出 `cuda: False`，说明 Notebook 没开 GPU，先到右侧设置里启用 GPU。

## 7. Smoke Test

先跑 2 条样本，确认模型能下载、输入路径正确、输出字段正常：

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

检查输出：

```bash
head -n 2 results/qwen25_15b_smoke_defense_run.jsonl
cat results/qwen25_15b_smoke_defense_summary.json
```

正常情况下，每条样本会有这些关键字段：

```text
attack_type
prompt_ppl
ppl_blocked
raw_success
ppl_defended_success
smooth_success
model_calls
elapsed_seconds
```

## 8. 正式 GCG + AutoDAN 对比实验

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

说明：

- `--limit 25` 表示每个攻击文件最多读取 25 条。
- 当前 GCG 有 25 条，AutoDAN 有 10 条，所以总共会评估 35 条。
- `--smooth-copies 5` 表示每条样本额外生成 5 个扰动版本。
- 每条样本大约需要 `1 + smooth_copies` 次生成，再加 PPL 计算。

## 9. 只计算 PPL/过滤指标

如果 Kaggle 时间紧，或者只想快速看 PPL filter 是否区分两类攻击，可以跳过生成：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl \
  --limit 25 \
  --skip-generation \
  --output results/qwen25_15b_ppl_only_defense_run.jsonl \
  --summary-output results/qwen25_15b_ppl_only_defense_summary.json
```

注意：`--skip-generation` 会使用输入文件已有的 `success`/`response` 字段，不会重新调用模型生成回答。

## 10. 显存不够时的参数

如果默认运行显存不够，可以按顺序尝试：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl \
  --limit 10 \
  --smooth-copies 3 \
  --max-new-tokens 48 \
  --output results/qwen25_15b_small_defense_run.jsonl \
  --summary-output results/qwen25_15b_small_defense_summary.json
```

如果仍然不够，再试 4bit：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl \
  --limit 25 \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --load-in-4bit \
  --device-map auto \
  --output results/qwen25_15b_4bit_defense_run.jsonl \
  --summary-output results/qwen25_15b_4bit_defense_summary.json
```

## 11. PPL 阈值

默认会用脚本内置 benign prompts 的第 95 分位数作为阈值。

如果有自己的 benign/dev prompts：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --benign-prompts data/benign_prompts.csv \
  --ppl-percentile 95 \
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl
```

也可以手动指定阈值：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --ppl-threshold 120 \
  --autodan-results results/autodan/autodan_final_normalized.jsonl \
  --gcg-results results/gcg_qwen25_15b.jsonl
```

报告中需要记录：

```text
ppl_threshold
ppl_threshold_source
```

## 12. 输出解读

每条 JSONL 结果包含：

- `attack_type`：`gcg` 或 `autodan`
- `prompt_ppl`：攻击 prompt 的困惑度
- `ppl_blocked`：是否被 PPL filter 拦截
- `raw_success`：无防御时 keyword-ASR 是否成功
- `ppl_defended_success`：PPL filter 后是否仍成功
- `smooth_success`：SemanticSmooth-lite 投票后是否仍成功
- `smooth_success_rate`：扰动副本中攻击成功比例
- `model_calls`：生成调用次数
- `elapsed_seconds`：该样本耗时

Summary 中重点看：

- `avg_ppl`
- `ppl_block_rate`
- `raw_asr`
- `ppl_defended_asr`
- `smooth_asr`
- `avg_model_calls`

## 13. 下载结果

Kaggle Notebook 跑完后，输出文件在：

```text
results/qwen25_15b_gcg_autodan_defense_run.jsonl
results/qwen25_15b_gcg_autodan_defense_summary.json
```

可以在 Notebook 右侧输出区下载，也可以复制到 `/kaggle/working/` 根目录：

```bash
cp results/qwen25_15b_gcg_autodan_defense_run.jsonl /kaggle/working/
cp results/qwen25_15b_gcg_autodan_defense_summary.json /kaggle/working/
```

## 14. 写报告时的提醒

这个项目是课程复现实验脚手架，不是完整 SemanticSmooth 论文训练管线。报告重点建议写：

- GCG 与 AutoDAN 的攻击形态差异。
- PPL filter 更容易拦截高 PPL 的 token suffix。
- AutoDAN 更自然，PPL filter 可能更难拦截。
- SemanticSmooth-lite 通过扰动和投票观察攻击稳定性。
- keyword-ASR 是轻量启发式指标，可能高估真实攻击成功率。
