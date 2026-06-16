# Defense Reproduction: Semantic Smoothing for Jailbreak Robustness

## 1. Scope of My Part

This section reports my part of the course project: reproducing a defense paper for large language model jailbreak robustness. The two attack reproductions, GCG and AutoDAN, are handled by other group members. In my experiment, their final attack outputs are used only as evaluation inputs; I do not inspect or reproduce the raw adversarial prompt contents. The focus is on whether lightweight defenses can reduce attack success on a small open-source model.

The defense paper selected for reproduction is:

> Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, and Shiyu Chang. *Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing*. arXiv:2402.16192, 2024.

The original paper proposes **SemanticSmooth**, a smoothing-based defense that applies semantic-preserving transformations to an input prompt, queries the model on multiple transformed copies, and aggregates the responses. Its main motivation is that jailbreak attacks may be unstable under meaning-preserving rewrites, while benign prompts should remain semantically stable.

## 2. Reproduction Setup

### 2.1 Model and Data

The target model in this reproduction is `Qwen/Qwen2.5-1.5B-Instruct`. The unified evaluation file contains 35 attack-positive samples:

| Attack Type | Number of Samples | Source Attack Success Labels |
| --- | ---: | ---: |
| GCG | 25 | 25 true, 0 false |
| AutoDAN | 10 | 6 true, 4 false |
| Total | 35 | 31 true, 4 false |

The data were normalized into `results/defense_training_inputs.jsonl`, and the formal defense run produced:

- `results/qwen25_15b_gcg_autodan_defense_run.jsonl`
- `results/qwen25_15b_gcg_autodan_defense_summary.json`

### 2.2 Defenses Implemented

I evaluated two prompt-level defenses.

First, I used a **PPL filter baseline**. For each input prompt, the script computes perplexity with the same Qwen model. The threshold is calibrated from built-in benign prompts using the 95th percentile. In this run, the threshold was:

| Threshold Source | PPL Threshold |
| --- | ---: |
| `builtin_benign_prompts:p95` | 195.1412 |

If the prompt perplexity is higher than this threshold, the prompt is blocked.

Second, I implemented **SemanticSmooth-lite**, a lightweight reproduction of the core SemanticSmooth idea. For each prompt, the script creates 5 transformed versions using simple semantic-preserving operations, such as removing wrapper-like text, compacting redundant wording, lightly reordering sentences, and substituting a small set of synonyms. The model is queried on the original prompt and on the 5 variants. The final defense outcome is decided by keyword-based refusal detection over the generated responses.

This implementation is intentionally simpler than the full SemanticSmooth paper. The original method uses a more systematic transformation set and can include a learned policy model for selecting transformations. Our reproduction keeps the central smoothing-and-aggregation idea but uses a low-cost course-project version.

### 2.3 Evaluation Metric

The main metric is keyword-based ASR. A response is counted as successful if it does not contain common refusal signals such as "I cannot", "I am sorry", or similar refusal phrases. This metric is easy to automate, but it can overestimate true harmful success because a non-refusal response is not always a detailed harmful answer.

## 3. Results

The defense summary is shown below.

| Attack | Defense | Avg PPL | Block Rate | ASR Before Defense | ASR After Defense | Extra Defense Generations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCG | PPL Filter | 12466.8011 | 84.00% | 76.00% | 16.00% | 0 |
| GCG | SemanticSmooth-lite | 12466.8011 | N/A | 76.00% | 76.00% | 5 |
| AutoDAN | PPL Filter | 13.0969 | 0.00% | 30.00% | 30.00% | 0 |
| AutoDAN | SemanticSmooth-lite | 13.0969 | N/A | 30.00% | 30.00% | 5 |

The combined evaluation script makes 6 generation calls per sample: 1 original generation plus 5 smoothing generations. Average runtime was 11.661 seconds per GCG sample and 12.755 seconds per AutoDAN sample.

## 4. Analysis

The clearest result is that the PPL filter strongly separates GCG-style attacks from AutoDAN-style attacks. GCG has an average PPL of 12466.8011, far above the threshold of 195.1412, so 84% of GCG samples are blocked. Its ASR drops from 76% before defense to 16% after PPL filtering. This supports the intuition that token-level adversarial suffixes often look unnatural to the language model and can be detected by a fluency-based filter.

AutoDAN behaves very differently. Its average PPL is only 13.0969, and none of the AutoDAN samples are blocked by the PPL filter. The ASR remains 30% before and after PPL filtering. This is consistent with the motivation of semantic jailbreak attacks: because the prompt is more fluent and natural-language-like, a detector based only on textual abnormality is not sufficient.

SemanticSmooth-lite did not reduce ASR in this run. For GCG, ASR remains 76%; for AutoDAN, ASR remains 30%. This does not necessarily contradict the SemanticSmooth paper, because our implementation is a simplified reproduction rather than the full method. The transformations used here are lightweight and rule-based, the sample size is small, and the aggregation criterion is keyword-based. In particular, the simplified transformations may not be strong enough to break the attack behavior, and keyword-ASR may count weak or irrelevant non-refusal answers as successful attacks.

The result is still useful: it shows that a naive smoothing implementation is not automatically effective. The defense depends heavily on transformation quality, aggregation rules, and the ability to preserve benign utility while disrupting adversarial intent.

## 5. Comparison with the Original Paper

The original SemanticSmooth paper argues that semantic-preserving transformations and response aggregation can improve robustness against attacks such as GCG, PAIR, and AutoDAN while maintaining nominal performance. Our reproduction follows the same high-level defense structure:

1. generate multiple transformed versions of the input,
2. query the target model on each version,
3. aggregate the resulting safety decisions.

However, several parts are simplified:

| Component | Original SemanticSmooth | Our Reproduction |
| --- | --- | --- |
| Transformations | richer semantic transformations | simple rule-based transformations |
| Transformation selection | may use adaptive policy selection | fixed lightweight heuristic |
| Evaluation scale | multiple models and benchmarks | 35 samples on Qwen2.5-1.5B-Instruct |
| Safety judgment | paper-level evaluation pipeline | refusal-keyword heuristic |
| Goal | robust defense evaluation | course-scale reproduction of core idea |

Therefore, our experiment should be interpreted as a partial reproduction of the defense mechanism, not a full reproduction of the paper's reported performance.

## 6. Limitations

There are four main limitations.

First, the sample size is small: 25 GCG samples and 10 AutoDAN samples. The results are useful for observing trends but not enough for a strong statistical conclusion.

Second, keyword-ASR is only a lightweight proxy. It may overestimate attack success when the model gives a non-refusal but harmless or vague answer.

Third, the PPL threshold is calibrated from built-in benign prompts rather than a large held-out benign set. A better reproduction should use a more representative benign dataset to estimate both false positive rate and robustness.

Fourth, SemanticSmooth-lite uses simple text transformations. The full SemanticSmooth method requires stronger semantic transformations and a more principled aggregation policy.

## 7. Conclusion

My defense reproduction shows a clear trade-off between simple fluency-based filtering and semantic-level jailbreak robustness. The PPL filter is highly effective against GCG in this experiment, reducing ASR from 76% to 16%, but it completely fails to block AutoDAN because AutoDAN prompts have low perplexity and resemble natural language. SemanticSmooth-lite reproduces the main idea of semantic smoothing, but in this lightweight implementation it does not reduce ASR, suggesting that transformation quality and aggregation design are critical for making smoothing defenses effective.

Overall, the experiment supports the central safety lesson of the project: defenses that only detect abnormal surface form are vulnerable to more fluent semantic jailbreaks, and robust defense requires reasoning over semantic stability rather than only prompt fluency.

## References

[1] Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, and Shiyu Chang. *Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing*. arXiv:2402.16192, 2024. https://arxiv.org/abs/2402.16192

[2] Alexander Robey, Eric Wong, Hamed Hassani, and George J. Pappas. *SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks*. arXiv:2310.03684, 2023. https://arxiv.org/abs/2310.03684
