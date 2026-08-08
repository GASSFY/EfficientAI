# MAS-Quant Local Artifacts

This document explains the directories and files produced when running MAS-Quant on Qwen2.5-VL (and similar MLLMs). **These paths are gitignored** unless noted otherwise; regenerate them locally as needed.

## Mental model

MAS-Quant uses **fake quantization**:

1. Load the original FP/BF16 Hugging Face model (large, usually under `~/.cache/huggingface/`).
2. Replace Linear layers with `QuantLinear` in memory.
3. Apply learned **smooth scales** from a small checkpoint.
4. On every forward, simulate low-bit weight/activation quantize–dequantize, then run float matmul.

There is **no** large “quantized 7B dump” under `outputs/` by default. The trainable product is primarily `mas_parameters.pth` (scales), not a full weight export. To optionally dump a fake-quantized model tree, pass `--save_dir` to `main.py`.

## Directory map

| Path | What it is | How to create | In git? |
|------|------------|---------------|---------|
| `act_scales/` | Per-layer activation max scales (text/vision/audio/all-in-one) for MAS init | `python generate_act_scale_shift.py --model ... --dataset-type text-vision --nsamples 128` | No |
| `cache/` | Serialized calibration dataloader (speed up re-runs) | Created automatically by Step1 / training when missing | No |
| `outputs/` | Per-run dirs: optional `mas_parameters.pth` + `log_rank0_*.txt` (+ eval `results/`) | `main.py --output_dir ./outputs ...` | No |
| `outputs_vl/` | Optional alternate output root (e.g. FP16 baselines) | Same as `outputs/`, different `--output_dir` | No |
| `logs/overnight/` | Aggregated logs from `run_overnight.sh` (`run_*.log`, `status_*.txt`, `summary_*.csv`) | `bash run_overnight.sh` | No |
| `experiments/*.md` | Compact human-readable score tables | Hand-maintained after eval | **Yes** |

### `act_scales/*.pt`

- Example: `act_scales/Qwen2.5-VL-7B-Instruct-text-vision-128.pt`
- Used when **training** MAS (`--epochs > 0` and no `--resume`).
- Not needed for eval-only (`--epochs 0 --resume .../mas_parameters.pth`).

### `cache/dataloader_*.cache`

- Cached calibration batches (`nsamples` items) for the given model + `dataset-type`.
- Safe to delete; the next run rebuilds it (slower first time).

### `outputs/<run_name>/`

Run name pattern (see `main.py`):

```text
{model}-{epochs}epochs-w{wbits}a{abits}-{postfix}-{timestamp}-{inference_mode}/
```

Important files:

- **`mas_parameters.pth`**: learned MAS smooth scales (and related LET params). Small (~5MB for 7B). This is the main quantization artifact to keep locally for resume/eval.
- **`log_rank0_*.txt`**: training/eval logs.
- **`results/`**: task-specific dumps (e.g. OCRBench breakdown) when the task writes them.

### `logs/` vs `outputs/`

- `outputs/`: written by `main.py` for each train/eval invocation.
- `logs/overnight/`: written by `run_overnight.sh` (tee of the whole overnight job + CSV summary).

## Helper scripts

| Script | Role |
|--------|------|
| `exp.sh` | Manual single-config train and/or eval (edit `TASKS`, `RESUME`, bits). |
| `run_overnight.sh` | Train missing bitwidths, then full eval suite; supports `SMOKE=1` for a tiny probe. |

### Generation length

`main.py` accepts `--gen_max_new_tokens` (default `16`):

- Use **16** for `mmmu_val` / `realworldqa` / `ai2d` (matches lmms-eval YAML).
- Use **128** for `ocrbench` (official default; shorter values can truncate OCR answers).

## Recommended local keep / delete

**Keep locally (not in git):**

- `outputs/*-2epochs-*/mas_parameters.pth` for configs you still evaluate.

**Safe to delete anytime:**

- `cache/`, `act_scales/`, `logs/`, `outputs/*-0epochs-*`, `__pycache__/`, `*.egg-info/`.

Regenerate scales with Step1; regenerate checkpoints with `epochs=2` training.
