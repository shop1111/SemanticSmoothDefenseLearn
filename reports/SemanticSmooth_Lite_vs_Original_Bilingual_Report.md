# Defense Experiment Report: SemanticSmooth-lite vs. Original-style SemanticSmooth

中文标题：SemanticSmooth-lite 与原版风格 SemanticSmooth 防御实验报告

## 1. Experimental Setup

This experiment evaluates prompt-level jailbreak defenses on a small open-source instruction model, `Qwen/Qwen2.5-1.5B-Instruct`. The evaluation set contains 50 attack-positive prompts in total: 25 GCG-style token-level suffix attacks and 25 AutoDAN initial-population prompts. The GCG samples represent high-perplexity adversarial suffix attacks, while the AutoDAN initial-population samples represent more fluent natural-language jailbreak prompts.

For safety, this report only discusses metadata, aggregate statistics, defense behavior, and failure modes. It does not reproduce or quote the original jailbreak prompts.

The experiment compares three defense outcomes:

| Defense | Description | Main Cost |
| --- | --- | ---: |
| PPL Filter | Blocks prompts whose perplexity exceeds a benign-calibrated threshold | No extra generation beyond scoring |
| SemanticSmooth-lite | Creates 5 rule-based semantic variants and aggregates response safety decisions | About 6 generation calls per sample |
| Original-style SemanticSmooth | Uses the model to generate 5 semantic rewrites, then queries the model on each rewrite | About 11 generation calls per sample |

The PPL threshold is calibrated from built-in benign prompts using the 95th percentile. The resulting threshold is `195.1412`.

中文说明：本实验使用同一批 50 条攻击样本比较三类防御输出：PPL 过滤、规则式 SemanticSmooth-lite，以及更接近原论文流程的模型改写版 SemanticSmooth。报告只呈现统计结果，不展示原始越狱提示词。

## 2. Defense Methods

### 2.1 PPL Filter

The PPL filter is a fluency-based baseline. It computes the perplexity of each input prompt using the same Qwen model. If the prompt perplexity is higher than the calibrated threshold, the prompt is blocked before generation. This method is expected to work well against token-level adversarial suffixes because such suffixes often look unnatural to the language model.

中文说明：PPL 过滤器主要捕捉“不自然”的 token 序列，因此理论上更适合拦截 GCG 这类 token-level suffix 攻击。

### 2.2 SemanticSmooth-lite

SemanticSmooth-lite is a lightweight reproduction of the smoothing idea. It creates 5 prompt variants using rule-based semantic transformations, such as compacting whitespace, removing wrapper-like text, lightly reordering sentences, and replacing a small set of words with synonyms. The model then generates responses to these variants, and the final result is decided by keyword-based refusal detection and majority voting.

This version is inexpensive compared with the original-style pipeline because it does not ask the model to rewrite the prompt before each smoothing query.

中文说明：lite 版本的核心特点是“规则式扰动”，不额外调用模型生成改写文本，因此成本较低，但语义变换能力有限。

### 2.3 Original-style SemanticSmooth

The original-style SemanticSmooth implementation is closer to the paper's main idea. For each input, the model first generates semantic-preserving rewrites using transformations including `spell_check`, `verb_tense`, `synonym`, `translate`, `summarize`, `paraphrase`, and `format`. The target model is then queried on each transformed prompt, and the responses are aggregated using keyword-based safety judgments.

This pipeline is more faithful to the original smoothing mechanism because the transformations are produced by a language model rather than simple rules. However, it is also more expensive: with 5 smoothing copies, each sample requires approximately 1 raw generation, 5 rewrite generations, and 5 response generations.

中文说明：original-style 版本更接近原论文，因为它使用模型生成语义保持改写；但它的代价也更高，每条样本大约需要 11 次生成调用。

## 3. Results

The main results are shown below. ASR means attack success rate under the corresponding defense outcome. Lower ASR is better.

| Attack Type | Avg PPL | Raw ASR | PPL Block Rate | PPL-defended ASR | Lite Smooth ASR | Original-style Smooth ASR | Lite Calls | Original Calls | Lite Seconds | Original Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCG | 12466.8011 | 0.76 | 0.84 | 0.16 | 0.76 | 0.24 | 6.0 | 11.0 | 11.448 | 19.997 |
| AutoDAN initial population | 115.4529 | 0.96 | 0.04 | 0.92 | 0.52 | 0.80 | 6.0 | 11.0 | 11.342 | 38.060 |

中文说明：GCG 的平均 PPL 极高，PPL 过滤和 original-style smoothing 都明显降低了 ASR；AutoDAN 的平均 PPL 低很多，PPL 基本拦不住，lite smoothing 在这组结果里反而比 original-style 更有效。

## 4. Analysis

### 4.1 GCG: PPL and original-style smoothing are effective

For GCG, the average prompt perplexity is `12466.8011`, far above the PPL threshold of `195.1412`. As a result, the PPL filter blocks 84% of GCG samples and reduces ASR from `0.76` to `0.16`. This confirms that high-perplexity adversarial suffixes are vulnerable to a simple fluency-based detector.

The original-style SemanticSmooth pipeline also performs well on GCG, reducing smooth ASR to `0.24`. This is a large improvement over SemanticSmooth-lite, whose ASR remains `0.76`. A likely explanation is that model-generated semantic rewrites can remove, dilute, or destabilize unnatural adversarial suffix patterns more effectively than rule-based perturbations.

中文说明：GCG 的失败条件很清楚：它依赖不自然的 suffix，因此 PPL 很容易识别；同时，模型改写也可能破坏 suffix 的精确 token 结构，所以 original-style smoothing 有效。

### 4.2 AutoDAN: natural-language attacks evade PPL, and lite outperforms original-style

AutoDAN initial-population prompts behave differently. Their average PPL is `115.4529`, below the threshold, so the PPL filter blocks only 4% of samples. PPL-defended ASR remains high at `0.92`, close to the raw ASR of `0.96`. This supports the intuition that natural-language jailbreak prompts are much harder to detect using fluency alone.

The smoothing results are more nuanced. SemanticSmooth-lite reduces AutoDAN ASR to `0.52`, while original-style SemanticSmooth only reduces it to `0.80`. This means that the more paper-like implementation is not automatically better for this attack type. One possible reason is that model-generated rewrites may preserve the high-level adversarial instruction too faithfully, especially when the prompt is already natural and fluent. In contrast, the simpler rule-based transformations may accidentally remove or disturb some wrapper structures used by AutoDAN prompts.

中文说明：AutoDAN 更像自然语言，因此 PPL 失效。original-style 改写虽然更“语义保持”，但这也可能保留了攻击意图；lite 的规则扰动反而可能破坏部分包装结构，所以在这组样本上更有效。

### 4.3 Cost-robustness trade-off

Original-style SemanticSmooth is substantially more expensive than the lite version. It uses about 11 generation calls per sample, compared with 6 calls for SemanticSmooth-lite. Runtime also increases: on GCG, average runtime rises from `11.448` seconds to `19.997` seconds per sample; on AutoDAN, it rises from `11.342` seconds to `38.060` seconds per sample.

This cost is justified for GCG because ASR decreases from `0.76` under lite smoothing to `0.24` under original-style smoothing. However, for AutoDAN, the extra cost does not produce a better result: ASR increases from `0.52` under lite smoothing to `0.80` under original-style smoothing.

中文说明：original-style 的成本明显更高，但收益只在 GCG 上成立；对 AutoDAN 来说，高成本并没有换来更强防御。

## 5. Limitations

First, the evaluation uses only 50 attack-positive samples, so the results should be interpreted as course-scale experimental evidence rather than a statistically complete benchmark.

Second, attack success is measured with keyword-based refusal detection. This metric is easy to automate, but it can overestimate success when a response is non-refusal but still harmless, incomplete, or vague.

Third, benign utility is not fully evaluated in these two runs. The PPL threshold is calibrated from benign prompts, but the report does not include a full false-positive and utility-retention comparison across the two smoothing methods.

Fourth, the original-style transformation model is the same small Qwen model used as the target model. A stronger paraphraser or a separately tuned transformation model may produce different results.

中文说明：主要局限包括样本量较小、keyword-ASR 粗糙、良性任务效用评估不足，以及改写模型能力有限。

## 6. Conclusion and Innovation Direction

The experiment shows that jailbreak defense effectiveness strongly depends on the attack form. PPL filtering is highly effective against GCG-style token suffix attacks, but it mostly fails against natural-language AutoDAN prompts. Original-style SemanticSmooth improves robustness against GCG compared with the lite version, but it is more expensive and less effective on AutoDAN in this run.

These findings suggest a practical innovation direction: an attack-aware adaptive defense. Instead of applying one defense uniformly, the system can first use PPL as a low-cost attack-shape signal. High-PPL prompts can be blocked or handled with a strict filter, while low-PPL natural-language prompts can be routed to smoothing-based defenses. The number and type of smoothing transformations can also be adjusted based on the prompt's risk profile, creating a cost-aware defense pipeline.

中文总结：本实验最重要的结论是“防御不能一刀切”。GCG 和 AutoDAN 的攻击形态不同，最有效的防御路径也不同。因此可以将后续创新点设计为 attack-aware adaptive defense：先用 PPL 判断攻击形态，再选择 PPL block、lite smoothing 或 original-style smoothing，从而同时考虑 ASR、防御成本和模型效用。

