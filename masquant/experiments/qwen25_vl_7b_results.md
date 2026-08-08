# Qwen2.5-VL-7B-Instruct MAS-Quant Results

Compact score table from a full (non-smoke) overnight run. Raw logs are not kept in git; see [ARTIFACTS.md](../ARTIFACTS.md).

## Setup

| Item | Value |
|------|--------|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Calibration | `text-vision`, `nsamples=128` |
| Training | `epochs=2`, `--let`, `--loss_multi_modal_mae_alpha`, `--symmetric`, `--group_size 0` |
| Inference mode | `split_scales` |
| Eval | `--epochs 0 --resume <mas_parameters.pth>`, `limit_multimodal=1.0` |
| `gen_max_new_tokens` | `16` for MMMU / RealWorldQA / AI2D; `128` for OCRBench |

## Scores

| Config | MMMU (val) | RealWorldQA | AI2D | OCRBench |
|--------|------------|-------------|------|----------|
| W4A16 | — | — | — | 0.714 |
| W8A8 | 0.4833 | 0.6928 | 0.8242 | 0.839 |
| W4A6 | 0.3400 | 0.5098 | 0.6606 | 0.642 |

Notes:

- W4A16 overnight job covered OCRBench only (other tasks not scheduled for that bitwidth in that run).
- Metrics follow lmms-eval: `mmmu_acc`, `exact_match` (RealWorldQA/AI2D), `ocrbench_accuracy`.
- Local checkpoints (gitignored): `outputs/Qwen2.5-VL-7B-Instruct-2epochs-w{4a16,4a8,8a8,4a6}-*-split_scales/mas_parameters.pth`.
