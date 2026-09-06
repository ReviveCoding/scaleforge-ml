# ScaleForge-ML Model Card

## Model and intended use

The selected model is the frozen pretrained `Qwen/Qwen2.5-1.5B-Instruct` checkpoint at revision `a8f3652d566d3a0035a6953a41da46833ba83686`, with tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It is evaluated as a compact English mathematical-reasoning workload and demonstration backend. It is not approved for consequential decisions, broad factual QA, or autonomous deployment.

Primary generation uses the frozen chat template, zero-shot reasoning prompt, `do_sample=false`, deterministic greedy decoding, explicit EOS/pad handling, maximum 512 new tokens for serving, and the versioned final-answer parser. M0 is the strongest development-selected pretrained baseline.

## Candidate selection

The compact LoRA program tested ranks 8/16/32, learning rates near 1e-4/2e-4, and attention-only versus attention+MLP targets using FIT/VALIDATION data. Small pilots eliminated poor candidates; only rank-32 attention+MLP at 1e-4 advanced. It used BF16, response-only loss, dynamic padding, microbatch 2 with accumulation 8, and global batch 16.

On all 723 development-validation examples the finalist scored 58.51% exact match versus 78.70% for M0 (-20.19 pp, paired 95% CI -24.20 to -16.18). It was nevertheless frozen as the sole challenger so the predeclared protected comparison could close the model question without cherry-picking.

## Protected evaluation

| Dataset | M0 | LoRA M* | Paired delta | Decision |
|---|---:|---:|---:|---|
| GSM8K test, n=1,319 | 66.64% | 49.43% | -17.21 pp (95% CI -20.24 to -14.18) | Reject M* |
| MATH-500, n=500 | 29.80% | 18.80% | -11.00 pp (95% CI -14.80 to -7.00) | OOD gate fails |

GSM8K transitions were 116 candidate-only wins, 343 baseline-only wins, 536 both correct, and 324 both wrong; exact McNemar p=4.23e-27. Failed generations count as incorrect. M0 had zero GSM8K parse failures and one conservative MATH-500 parse failure.

## Decision and limitations

MODEL is PASS with M0 retained; the LoRA challenger is BLOCK. Fine-tuning complexity is not adopted when it harms both primary and OOD metrics. SF-MODEL-v3 is an evaluator-only rescore of complete, immutable v2 text after two protected parser defects. Its text is not an independent third generation run. Exact matching is not symbolic equivalence and cannot show reasoning faithfulness. Results apply only to these revisions, prompts, parser, decoding controls, and datasets.

Canonical evidence: `FREEZE_MANIFEST.json`, `FINAL_ACCESS_LEDGER.json`, `artifacts/analysis/model/model_qualification_sf_model_v3.json`, and `artifacts/warehouse/model_predictions.parquet`.
