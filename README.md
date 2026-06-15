# SemanticSmooth Defense Learning Project

这是一个面向 Kaggle 的轻量复现实验项目，用来观察 PPL filter 和
SemanticSmooth-lite 防御在两类 jailbreak attack 上的表现差异：

- **GCG 类攻击**：token-level adversarial suffix，常表现为不自然的 token 后缀，PPL 通常较高。
- **AutoDAN 类攻击**：natural-language jailbreak prompt，可读性更强，语义更自然，PPL 通常更低。

默认模型是 `Qwen/Qwen2.5-1.5B-Instruct`。

## 项目结构

```text
src/
  ppl_smoothing_defense_kaggle.py        # PPL filter + SemanticSmooth-lite 评估入口
  prepare_defense_training_inputs.py     # 归一化 GCG/AutoDAN 原始结果
reports/
  Kaggle_Qwen25_15B_PPL_Smoothing运行说明.md
results/
  gcg_qwen25_15b.jsonl                   # 正式 GCG 输入，25 条
  autodan/
    autodan_final_normalized.jsonl        # 正式 AutoDAN 输入，10 条
  defense_training_inputs.jsonl           # 合并后的 attack-positive 输入，35 条
  defense_training_inputs_summary.json
  defense_training_inputs_summary.csv
  archive/                                # 旧输入和中间生成产物归档
examples/
  README.md
requirements.txt
```

正式 Kaggle 运行主要使用：

- `results/gcg_qwen25_15b.jsonl`
- `results/autodan/autodan_final_normalized.jsonl`

`results/archive/qwen25_3b_hga_50_legacy_hga.jsonl` 只是旧输入归档。
AutoDAN 生成过程中的中间文件放在 `results/archive/autodan_generation/`。

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

先跑 2 条样本，确认模型下载、输入路径和输出文件都正常：

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

查看结果：

```bash
head -n 2 results/qwen25_15b_smoke_defense_run.jsonl
cat results/qwen25_15b_smoke_defense_summary.json
```

## 正式 GCG + AutoDAN 对比实验

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
- 当前输入会实际评估 25 条 GCG 和 10 条 AutoDAN，共 35 条。
- `--smooth-copies 5` 表示每条样本额外生成 5 个扰动 prompt。
- 每条样本大约需要 1 次原始生成 + 5 次平滑扰动生成，再加 PPL 计算。

## 只跑 PPL/过滤指标

如果 Kaggle 时间紧，可以先跳过生成，只计算 PPL filter 相关指标：

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

## 显存不够时

先减小样本数、平滑副本数和生成长度：

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

如果仍然显存不足，再尝试 4bit 加载：

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

## 重新生成输入

如果你有新的 GCG 或 AutoDAN 原始结果，可以用下面命令归一化：

```bash
python src/prepare_defense_training_inputs.py \
  --gcg-artifact /path/to/GCG/Qwen2.5-1.5B-Instruct.json \
  --gcg-eval /path/to/eval_local_details.json \
  --autodan /path/to/autodan_final.jsonl
```

默认输出：

- `results/gcg_qwen25_15b.jsonl`
- `results/autodan/autodan_final_normalized.jsonl`
- `results/defense_training_inputs.jsonl`
- `results/defense_training_inputs_summary.json`
- `results/defense_training_inputs_summary.csv`

当前 `defense_training_inputs.jsonl` 只有攻击样本，适合做防御评估输入或后续扩展训练集。如果要训练二分类防御器，还需要加入 benign/refusal/ordinary prompts。

## 主要输出字段

- `attack_type`：攻击类型，`gcg` 或 `autodan`。
- `prompt_ppl`：攻击 prompt 的 perplexity。
- `ppl_blocked`：是否被 PPL filter 拦截。
- `raw_success`：无防御时 keyword-ASR 是否成功。
- `ppl_defended_success`：经过 PPL filter 后攻击是否仍成功。
- `smooth_success`：SemanticSmooth-lite 投票后攻击是否仍成功。
- `smooth_success_rate`：扰动副本中攻击成功的比例。
- `model_calls`：本脚本对模型的生成调用次数。
- `elapsed_seconds`：单条样本耗时。

## 推荐报告表格

| 攻击 | 防御 | 平均 PPL | 拦截率 | 防御前 ASR | 防御后 ASR | 平均调用次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCG | PPL Filter |  |  |  |  |  |
| GCG | SemanticSmooth-lite |  |  |  |  |  |
| AutoDAN | PPL Filter |  |  |  |  |  |
| AutoDAN | SemanticSmooth-lite |  |  |  |  |  |

## 注意

这个项目是课程复现实验脚手架，不是完整 SemanticSmooth 论文训练管线。重点是比较简单 PPL baseline 与 smoothing-style prompt perturbation 的效果，并解释为什么自然语言语义攻击通常比 token-level suffix 攻击更难被简单规则拦住。
