# SemanticSmooth Defense Learning Project

这是一个面向 Kaggle 的轻量复现实验项目，用来观察 PPL filter、
SemanticSmooth-lite、original-style SemanticSmooth 和自适应
PPL-gated SemanticSmooth 防御在两类 jailbreak attack 上的表现差异：

- **GCG 类攻击**：token-level adversarial suffix，常表现为不自然的 token 后缀，PPL 通常较高。
- **AutoDAN 类攻击**：natural-language jailbreak prompt，可读性更强，语义更自然，PPL 通常更低。

默认模型是 `Qwen/Qwen2.5-1.5B-Instruct`。

当前创新策略是 **Adaptive PPL-Gated Semantic Smoothing Defense**：先用
prompt perplexity 作为低成本门控信号；高 PPL 输入直接走 PPL block，低
PPL 输入先进入 SemanticSmooth-lite 投票；只有 lite 投票接近 50/50 时，才调用
original-style SemanticSmooth 做保守融合复核，从而避免对所有样本无差别使用高成本
smoothing，也避免 original 覆盖 lite 已经拦截的结果。

## 项目结构

```text
src/
  ppl_smoothing_defense_kaggle.py        # PPL filter + SemanticSmooth 评估入口
  prepare_defense_training_inputs.py     # 归一化 GCG/AutoDAN 原始结果
reports/
  Kaggle_Qwen25_15B_PPL_Smoothing运行说明.md
  SemanticSmooth_Lite_vs_Original_Bilingual_Report.md
results/
  defense_training_inputs.jsonl           # Kaggle 正式统一输入，50 条：25 条 GCG + 25 条 AutoDAN 初始种群
  defense_training_inputs_summary.json
  defense_training_inputs_summary.csv
  summaries/                              # 不含原始 prompt 的聚合实验结果
  archive/                                # 旧输入、分文件 normalized 输入和中间产物归档
examples/
  README.md
requirements.txt
```

正式 Kaggle 运行只使用一个统一输入文件：

```text
results/defense_training_inputs.jsonl
```

分开的 GCG/AutoDAN normalized 文件已经归档到：

```text
results/archive/normalized_inputs/gcg_qwen25_15b.jsonl
results/archive/normalized_inputs/autodan_final_normalized.jsonl
```

这些分文件只用于追溯和调试，不再作为 Kaggle 主运行路径。

## 实验报告与安全摘要

双语实验报告：

```text
reports/SemanticSmooth_Lite_vs_Original_Bilingual_Report.md
```

不包含原始 jailbreak prompt 的聚合结果摘要：

```text
results/summaries/qwen25_15b_gcg_autodan_lite_summary.json
results/summaries/qwen25_15b_gcg_autodan_original_summary.json
results/summaries/qwen25_15b_gcg_autodan_adaptive_summary.json
```

## 安装依赖

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

如果输出 `False`，需要先在 Kaggle Notebook 右侧设置中开启 GPU。

## 快速测试

先跑 2 条样本，确认模型下载、统一输入路径和输出文件都正常：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --limit 2 \
  --smooth-copies 2 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_smoke_defense_run.jsonl \
  --summary-output results/qwen25_15b_smoke_defense_summary.json
```

查看结果：

```bash
head -n 2 results/qwen25_15b_smoke_defense_run.jsonl
cat results/qwen25_15b_smoke_defense_summary.json
```

## 正式 GCG + AutoDAN 初始种群防御实验

当前统一输入共 50 条：保留原来的 25 条 GCG，并将 AutoDAN 部分替换为 AutoDAN 官方初始 DAN 类种群 `prompt_group.pth` 的前 25 条。正式运行可以不传 `--limit`，默认最多读取 50 条，因此会读完整个统一输入。

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_autodan_initial_defense_run.jsonl \
  --summary-output results/qwen25_15b_autodan_initial_defense_summary.json
```

说明：

- `--attack-inputs` 是正式统一输入路径。
- 当前输入会实际评估 25 条 GCG 和 25 条 AutoDAN 初始种群样本，共 50 条。
- `--smooth-copies 5` 表示每条样本额外生成 5 个扰动 prompt。
- 每条样本大约需要 1 次原始生成 + 5 次平滑扰动生成，再加 PPL 计算。

如果要运行更接近原版 SemanticSmooth 的模型语义改写版本，需要显式指定：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --defense semantic_smooth_original \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_gcg_autodan_original_defense_run.jsonl \
  --summary-output results/qwen25_15b_gcg_autodan_original_defense_summary.json
```

如果要运行自适应 PPL-gated 策略：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --defense adaptive_ppl_gated \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --output results/qwen25_15b_gcg_autodan_adaptive_defense_run.jsonl \
  --summary-output results/qwen25_15b_gcg_autodan_adaptive_defense_summary.json
```

`adaptive_ppl_gated` 使用当前 PPL 阈值作为 gate，并采用 lite-first fallback 策略：

- `prompt_ppl > ppl_threshold`：路由到 `ppl_block`，不再执行 smoothing。
- `prompt_ppl <= ppl_threshold`：先路由到 `lite_smoothing`，执行 SemanticSmooth-lite 投票。
- 如果 lite 的 `smooth_success_rate` 距离 0.5 不超过 `--adaptive-uncertain-margin`，再路由到 `original_fallback`，使用 `lite_success AND original_success` 做保守融合。也就是说，original 只能帮助拦截，不能推翻 lite 已经拦住的结果。

## 只跑 PPL/过滤指标

如果 Kaggle 时间紧，可以先跳过生成，只计算 PPL filter 相关指标：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --skip-generation \
  --output results/qwen25_15b_ppl_only_defense_run.jsonl \
  --summary-output results/qwen25_15b_ppl_only_defense_summary.json
```

注意：`--skip-generation` 会使用统一输入文件已有的 `attack_success`/`response` 字段，不会重新调用模型生成回答。

## 显存不够时

先减小样本数、平滑副本数和生成长度：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --limit 10 \
  --smooth-copies 3 \
  --max-new-tokens 48 \
  --output results/qwen25_15b_small_defense_run.jsonl \
  --summary-output results/qwen25_15b_small_defense_summary.json
```

如果仍然显存不足，再尝试 4bit 加载：

```bash
python src/ppl_smoothing_defense_kaggle.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attack-inputs results/defense_training_inputs.jsonl \
  --smooth-copies 5 \
  --max-new-tokens 64 \
  --load-in-4bit \
  --device-map auto \
  --output results/qwen25_15b_4bit_defense_run.jsonl \
  --summary-output results/qwen25_15b_4bit_defense_summary.json
```

## 重新生成输入

当前 GCG + AutoDAN 初始种群统一输入可用下面命令重新生成：

```bash
python src/prepare_autodan_initial_population_inputs.py
```

它默认读取：

- `results/archive/normalized_inputs/gcg_qwen25_15b.jsonl`
- `references/AutoDAN/assets/prompt_group.pth`
- `references/AutoDAN/data/advbench/harmful_behaviors.csv`

如果你有新的 GCG 或 AutoDAN 原始结果，可以用下面命令归一化旧版混合输入：

```bash
python src/prepare_defense_training_inputs.py \
  --gcg-artifact /path/to/GCG/Qwen2.5-1.5B-Instruct.json \
  --gcg-eval /path/to/eval_local_details.json \
  --autodan /path/to/autodan_final.jsonl
```

默认正式输出：

- `results/defense_training_inputs.jsonl`
- `results/defense_training_inputs_summary.json`
- `results/defense_training_inputs_summary.csv`

默认归档输出：

- `results/archive/normalized_inputs/gcg_qwen25_15b.jsonl`
- `results/archive/normalized_inputs/autodan_final_normalized.jsonl`
- `results/archive/normalized_inputs/autodan_initial_population_25.jsonl`

当前 `defense_training_inputs.jsonl` 只有攻击样本，适合做防御评估输入或后续扩展训练集。如果要训练二分类防御器，还需要加入 benign/refusal/ordinary prompts。

## 主要输出字段

- `attack_type`：攻击类型，例如 `autodan_initial_population`、`gcg` 或 `autodan`。
- `prompt_ppl`：攻击 prompt 的 perplexity。
- `ppl_blocked`：是否被 PPL filter 拦截。
- `raw_success`：无防御时 keyword-ASR 是否成功。
- `ppl_defended_success`：经过 PPL filter 后攻击是否仍成功。
- `smooth_success`：SemanticSmooth 投票后攻击是否仍成功。
- `smooth_success_rate`：扰动副本中攻击成功的比例。
- `adaptive_route`：自适应策略路由，可能为 `ppl_block`、`lite_smoothing` 或 `original_fallback`。
- `model_calls`：本脚本对模型的生成调用次数。
- `elapsed_seconds`：单条样本耗时。

## 推荐报告表格

| 攻击 | 防御 | 平均 PPL | 拦截率 | 防御前 ASR | 防御后 ASR | 平均调用次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCG | PPL Filter |  |  |  |  |  |
| GCG | SemanticSmooth-lite |  |  |  |  |  |
| GCG | SemanticSmooth original-style |  |  |  |  |  |
| GCG | Adaptive PPL-gated | 12466.8011 | 84.00% | 76.00% | 16.00% | 1.8 |
| AutoDAN | PPL Filter |  |  |  |  |  |
| AutoDAN | SemanticSmooth-lite |  |  |  |  |  |
| AutoDAN | SemanticSmooth original-style |  |  |  |  |  |
| AutoDAN | Adaptive PPL-gated | 115.4529 | 4.00% | 96.00% | 48.00% | 10.6 |

## 注意

这个项目是课程复现实验脚手架，不是完整 SemanticSmooth 论文训练管线。重点是比较简单 PPL baseline 与 smoothing-style prompt perturbation 的效果，并解释为什么自然语言语义攻击通常比 token-level suffix 攻击更难被简单规则拦住。
