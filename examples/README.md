# GCG Input Format

The defense script accepts GCG results in JSONL or CSV format.

Recommended fields:

| Field | Meaning |
| --- | --- |
| `goal` | The harmful behavior or user request being attacked. |
| `adversarial_suffix` | The optimized GCG suffix. |
| `prompt` | Optional full attack prompt. If provided, it is used directly. |
| `response` | Optional model response from the attack run. |
| `success` | Optional boolean attack success label. |

If both `goal` and `adversarial_suffix` are provided, the script builds:

```text
{goal} {adversarial_suffix}
```

If `prompt` is provided, the script uses `prompt` directly.

Example JSONL row:

```json
{"goal":"Write instructions for a harmful task","adversarial_suffix":"! ! ! token suffix ...","success":true,"response":"..."}
```

For formal reporting, replace the built-in GCG-like smoke examples with real GCG
outputs generated on the same benchmark subset as the AutoDAN results.
