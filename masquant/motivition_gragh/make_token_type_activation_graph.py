#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token-type activation observation figure for AdaMAS.

A single image-text input is fed into Qwen2.5-VL-3B-Instruct. Tokens are grouped
by semantic/source type:
  - text prompt tokens
  - text question/body tokens
  - visual text-region tokens
  - visual subject-region tokens
  - visual background-region tokens

The figure reports hidden activation statistics for each group. This supports
the observation that different token types, and different tokens within the
same visual modality, have different numerical characteristics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL_DIR = "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/EfficientAI-main/masquant/motivition_gragh"

COL = {
    "prompt": "#4C78A8",
    "body": "#72B7B2",
    "visual_text": "#E15759",
    "subject": "#59A14F",
    "background": "#8DA0CB",
    "dark": "#222A35",
    "muted": "#65717F",
    "grid": "#DDE3EA",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_input_image(path: str, size: int = 448) -> Tuple[Image.Image, Dict[str, Tuple[int, int, int, int]]]:
    im = Image.new("RGB", (size, size), (244, 247, 251))
    d = ImageDraw.Draw(im)
    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 34)
        font_mid = ImageFont.truetype("DejaVuSans.ttf", 24)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_region = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font_big = font_mid = font_small = font_region = None

    boxes = {
        "background": (24, 36, 184, 402),
        "visual_text": (220, 48, 410, 155),
        "subject": (230, 225, 398, 398),
    }

    # Background/context area.
    d.rectangle(boxes["background"], fill=(195, 207, 226), outline=(119, 139, 169), width=2)
    for x in [48, 88, 128, 168]:
        d.line([x, 52, x - 28, 390], fill=(176, 188, 207), width=2)
    d.text((30, 405), "background", fill=(75, 88, 110), font=font_region)

    # Subject area.
    d.rectangle(boxes["subject"], fill=(255, 255, 255), outline=(151, 160, 170), width=2)
    d.ellipse([260, 260, 368, 368], fill=(86, 158, 109), outline=(45, 110, 72), width=3)
    d.rectangle([290, 235, 338, 275], fill=(245, 184, 72), outline=(146, 108, 41), width=2)
    d.text((268, 402), "subject", fill=(57, 126, 85), font=font_region)

    # Visual text/OCR region.
    d.rectangle(boxes["visual_text"], fill=(255, 255, 255), outline=(170, 170, 170), width=2)
    d.text((232, 60), "LABEL", fill=(20, 20, 20), font=font_big)
    d.text((232, 108), "ID: 3A-19", fill=(20, 20, 20), font=font_mid)
    d.text((246, 158), "visual text", fill=(160, 55, 55), font=font_region)

    im.save(path)
    return im, boxes


def prepare_inputs(processor: AutoProcessor, image: Image.Image, instruction: str, question: str) -> Dict[str, torch.Tensor]:
    user_text = instruction + "\n" + question
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs["_prompt_string"] = prompt
    return inputs


def find_subsequence(seq: List[int], sub: List[int]) -> Tuple[int, int] | None:
    if not sub:
        return None
    for i in range(0, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i, i + len(sub)
    return None


def box_to_visual_token_mask(box: Tuple[int, int, int, int], image_size: Tuple[int, int], n_visual_tokens: int) -> np.ndarray:
    side = int(round(math.sqrt(n_visual_tokens)))
    if side * side != n_visual_tokens:
        # Fallback to a near-square layout.
        side = int(math.floor(math.sqrt(n_visual_tokens)))
        other = int(math.ceil(n_visual_tokens / max(side, 1)))
    else:
        other = side
    img_w, img_h = image_size
    yy, xx = np.meshgrid(np.arange(side), np.arange(other), indexing="ij")
    cx = (xx + 0.5) / other * img_w
    cy = (yy + 0.5) / side * img_h
    x1, y1, x2, y2 = box
    mask = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return mask.reshape(-1)[:n_visual_tokens]


def symmetric_fake_quant(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    x = x.float()
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def occlude_region(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    out = image.copy()
    d = ImageDraw.Draw(out)
    x1, y1, x2, y2 = box
    d.rectangle([x1, y1, x2, y2], fill=(218, 222, 228), outline=(120, 128, 138), width=2)
    for x in range(x1 + 8, x2, 18):
        d.line([x, y1 + 6, x - 24, y2 - 6], fill=(198, 203, 210), width=2)
    return out


def answer_nll(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    instruction: str,
    question: str,
    answer: str,
) -> float:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction + "\n" + question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt + answer
    inputs = processor(text=[full_text], images=[image], return_tensors="pt")
    model_inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    tokenizer = processor.tokenizer
    input_ids = inputs["input_ids"][0].tolist()
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    span = find_subsequence(input_ids, answer_ids)
    if span is None:
        # Fallback: use the final answer-length tokens.
        span = (len(input_ids) - len(answer_ids), len(input_ids))

    labels = torch.full_like(model_inputs["input_ids"], -100)
    labels[0, span[0]:span[1]] = model_inputs["input_ids"][0, span[0]:span[1]]
    with torch.no_grad():
        out = model(**model_inputs, labels=labels, return_dict=True)
    return float(out.loss.detach().float().cpu())


def compute_ablation_impact(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    boxes: Dict[str, Tuple[int, int, int, int]],
    instruction: str,
    question: str,
) -> Dict[str, float]:
    answer = " 3A-19, green object"
    base = answer_nll(model, processor, image, instruction, question, answer)
    variants = {
        "Prompt tokens": (image, "Instruction:", question),
        "Question tokens": (image, instruction, "Question:"),
        "Visual text": (occlude_region(image, boxes["visual_text"]), instruction, question),
        "Subject region": (occlude_region(image, boxes["subject"]), instruction, question),
        "Background": (occlude_region(image, boxes["background"]), instruction, question),
    }
    impact = {}
    signed_delta = {}
    for name, (img_i, inst_i, ques_i) in variants.items():
        nll = answer_nll(model, processor, img_i, inst_i, ques_i, answer)
        delta = nll - base
        signed_delta[name] = delta
        impact[name] = abs(delta)
    return {"baseline_nll": base, "impact": impact, "signed_delta": signed_delta, "target_answer": answer.strip()}


def group_stats(hidden: torch.Tensor, token_indices: np.ndarray) -> Tuple[float, float, float]:
    if len(token_indices) == 0:
        return float("nan"), float("nan"), float("nan")
    h = hidden[token_indices].float()
    per_token_mag = h.abs().mean(dim=-1)
    magnitude = float(per_token_mag.mean().cpu())
    cv = float((per_token_mag.std() / per_token_mag.mean().clamp_min(1e-8)).cpu())
    q = symmetric_fake_quant(h, bits=4)
    qerr = float((h - q).abs().mean().cpu())
    return magnitude, cv, qerr


def compute_stats(model_dir: str, output_dir: str, device: str = "auto", layer_mode: str = "middle") -> Dict:
    ensure_dir(output_dir)
    image_path = os.path.join(output_dir, "token_type_activation_input.png")
    image, boxes = create_input_image(image_path)

    processor = AutoProcessor.from_pretrained(model_dir)
    instruction = "Instruction: Answer the visual question using the label and the main object."
    question = "Question: What ID is written on the label, and what is the main object?"
    inputs = prepare_inputs(processor, image, instruction, question)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    model_inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items() if not k.startswith("_")}
    with torch.no_grad():
        out = model(**model_inputs, output_hidden_states=True, return_dict=True)

    hidden_states = out.hidden_states[1:]
    if layer_mode == "last":
        h = hidden_states[-1][0].detach().cpu()
        layer_desc = f"last layer ({len(hidden_states)-1})"
    elif layer_mode == "mean":
        h = torch.stack([x[0].detach().cpu().float() for x in hidden_states], dim=0).mean(dim=0)
        layer_desc = "mean over decoder layers"
    else:
        mid = len(hidden_states) // 2
        h = hidden_states[mid][0].detach().cpu()
        layer_desc = f"middle layer ({mid})"

    input_ids = inputs["input_ids"][0].tolist()
    mm_type = inputs.get("mm_token_type_ids", None)
    if mm_type is not None:
        vision_positions = torch.where(mm_type[0].bool())[0].cpu().numpy()
        text_positions = torch.where(~mm_type[0].bool())[0].cpu().numpy()
    else:
        vision_positions = np.where(np.asarray(input_ids) == 151655)[0]
        text_positions = np.asarray([i for i in range(len(input_ids)) if i not in set(vision_positions)])

    tokenizer = processor.tokenizer
    instruction_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    instr_span = find_subsequence(input_ids, instruction_ids)
    ques_span = find_subsequence(input_ids, question_ids)
    if instr_span is None:
        prompt_indices = text_positions[:max(1, len(text_positions)//3)]
    else:
        prompt_indices = np.arange(instr_span[0], instr_span[1])
    if ques_span is None:
        body_indices = text_positions[-max(1, len(text_positions)//3):]
    else:
        body_indices = np.arange(ques_span[0], ques_span[1])

    n_vis = len(vision_positions)
    visual_masks = {
        name: box_to_visual_token_mask(box, image.size, n_vis)
        for name, box in boxes.items()
    }
    groups = {
        "Prompt tokens": prompt_indices,
        "Question tokens": body_indices,
        "Visual text": vision_positions[visual_masks["visual_text"]],
        "Subject region": vision_positions[visual_masks["subject"]],
        "Background": vision_positions[visual_masks["background"]],
    }
    metrics = np.array([group_stats(h, idx) for idx in groups.values()], dtype=np.float64)
    ablation = compute_ablation_impact(model, processor, image, boxes, instruction, question)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "image_path": image_path,
        "boxes": boxes,
        "groups": list(groups.keys()),
        "metrics": ["Activation magnitude", "CV", "W4 act. error"],
        "values": metrics,
        "ablation_baseline_nll": ablation["baseline_nll"],
        "ablation_target_answer": ablation["target_answer"],
        "ablation_impact": [ablation["impact"][g] for g in groups.keys()],
        "ablation_signed_delta": [ablation["signed_delta"][g] for g in groups.keys()],
        "layer_desc": layer_desc,
        "instruction": instruction,
        "question": question,
    }


def normalize_cols(x: np.ndarray) -> np.ndarray:
    y = x.copy().astype(np.float64)
    for j in range(y.shape[1]):
        col = y[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        y[:, j] = 0 if hi - lo < 1e-12 else (col - lo) / (hi - lo)
    return y


def normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def smooth_curve(x: np.ndarray, passes: int = 2) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    for _ in range(passes):
        y = np.pad(y, (1, 1), mode="edge")
        y = np.convolve(y, kernel, mode="valid")
    return y


def load_layer_evolution(output_dir: str) -> Dict[str, List[float]] | None:
    path = os.path.join(output_dir, "motivation_stats_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["layers", "text_error_proxy", "vision_error_proxy"]
    if not all(k in data for k in required):
        return None
    return {k: data[k] for k in required}


def draw_figure(stats: Dict, output_dir: str, output_prefix: str = "adamas_token_type_activation_graph") -> Tuple[str, str]:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "font.size": 7.0,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(3.55, 3.72), facecolor="white")
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.42, 0.90],
        left=0.055, right=0.975, top=0.955, bottom=0.105,
        hspace=0.30,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.40, 0.60], wspace=0.075)
    ax_input = fig.add_subplot(top[0, 0])
    ax_bar = fig.add_subplot(top[0, 1])
    ax_curve = fig.add_subplot(outer[1, 0])

    coral = "#B96A62"
    mountain_blue = "#6E8FAE"
    morandi_green = "#8BA888"
    soft_gray = "#C8C8C2"
    slate = "#5B6472"
    grid_col = "#E8E8E3"
    axis_col = "#40464F"
    box_edge = "#D9D6CF"
    bg_wash = "#FAFAF7"

    # Panel title placed outside the drawing area.
    fig.text(0.055, 0.984, "(a) Token-level Sensitivity Variance", ha="left", va="top",
             fontweight="bold", fontsize=8.3, color=axis_col)

    # ------------------------------------------------------------------
    # Compact multimodal input schematic, left 40% of panel (a).
    # ------------------------------------------------------------------
    ax_input.set_xlim(0, 1)
    ax_input.set_ylim(0, 1)
    ax_input.axis("off")
    ax_input.add_patch(Rectangle((0.00, 0.00), 1.00, 0.90, facecolor=bg_wash, edgecolor=box_edge, linewidth=0.65))

    # Seamless text/question strips.
    ax_input.add_patch(Rectangle((0.030, 0.755), 0.94, 0.105, facecolor="#FFFFFF", edgecolor="#DDDAD2", linewidth=0.45))
    ax_input.add_patch(Rectangle((0.030, 0.650), 0.94, 0.105, facecolor="#FFFFFF", edgecolor="#DDDAD2", linewidth=0.45))
    ax_input.text(0.055, 0.807, "Prompt: answer visually", fontsize=5.35, color=slate, ha="left", va="center")
    ax_input.text(0.055, 0.702, "Question: read label", fontsize=5.35, color=slate, ha="left", va="center")

    # Compact image thumbnail immediately below the text strips.
    image_path = stats.get("image_path")
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
        ax_img = ax_input.inset_axes([0.020, 0.030, 0.96, 0.600])
        ax_img.imshow(image)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#D6D2CB")
            spine.set_linewidth(0.45)
        region_colors = {"visual_text": coral, "subject": morandi_green, "background": mountain_blue}
        for name, c in region_colors.items():
            if name in stats.get("boxes", {}):
                x1, y1, x2, y2 = stats["boxes"][name]
                ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=0.95, alpha=0.95))
    else:
        ax_input.add_patch(Rectangle((0.020, 0.030), 0.96, 0.600, facecolor="#FFFFFF", edgecolor="#D6D2CB", linewidth=0.45))
        ax_input.text(0.50, 0.33, "Image", ha="center", va="center", fontsize=6, color=slate)

    # ------------------------------------------------------------------
    # Compact sensitivity bars, right 60% of panel (a).
    # ------------------------------------------------------------------
    groups = stats["groups"]
    impacts = np.asarray(stats["ablation_impact"], dtype=np.float64)
    impact_map = {g: float(v) for g, v in zip(groups, impacts)}
    ordered_groups = ["Visual text", "Question tokens", "Subject region", "Prompt tokens", "Background"]
    ordered_impacts = np.array([impact_map[g] for g in ordered_groups], dtype=np.float64)
    colors = [coral, mountain_blue, morandi_green, soft_gray, soft_gray]
    y = np.arange(len(ordered_groups))
    ax_bar.barh(y, ordered_impacts, color=colors, height=0.36, edgecolor="white", linewidth=0.55)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_ylim(len(ordered_groups) - 0.35, -0.55)
    ax_bar.set_xlabel(r"Sensitivity after masking ($\Delta$ NLL)", fontsize=6.8, labelpad=1.5)
    ax_bar.grid(axis="x", color=grid_col, linewidth=0.5)
    ax_bar.set_axisbelow(True)
    xmax = max(float(np.nanmax(ordered_impacts)) * 1.13, 2.28)
    ax_bar.set_xlim(0, xmax)
    ax_bar.set_xticks([0, 1, 2])
    ax_bar.axvline(0, color="#DADDD9", linewidth=0.55, zorder=1)
    for i, (name, val) in enumerate(zip(ordered_groups, ordered_impacts)):
        label_y = i - 0.205
        ax_bar.text(0.03, label_y, name, va="bottom", ha="left", fontsize=5.85, color="#3F3F3B")
        num_x = val + xmax * 0.010
        ax_bar.text(num_x, i, f"{val:.2f}", va="center", ha="left", fontsize=6.25, color=axis_col)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["bottom"].set_color("#BFBDB7")
    ax_bar.tick_params(axis="y", length=0, pad=1.0, colors=axis_col)
    ax_bar.tick_params(axis="x", colors=axis_col, labelsize=6.4, length=2.0, pad=1.2)

    # ------------------------------------------------------------------
    # Panel (b): compressed layer-wise modality error evolution.
    # ------------------------------------------------------------------
    ax_curve.text(0.0, 1.015, "(b) Layer-wise Modality Error Evolution", transform=ax_curve.transAxes,
                  ha="left", va="bottom", fontweight="bold", fontsize=8.3, color=axis_col, clip_on=False)
    layer_stats = stats.get("layer_evolution") or load_layer_evolution(output_dir)
    if layer_stats is None:
        ax_curve.text(0.5, 0.5, "Layer-wise cache not found", ha="center", va="center", fontsize=7.0, color="#6B7280")
        ax_curve.set_axis_off()
    else:
        layers = np.asarray(layer_stats["layers"], dtype=np.float64)
        text_err = smooth_curve(normalize_vector(np.asarray(layer_stats["text_error_proxy"], dtype=np.float64)), passes=3)
        vision_err = smooth_curve(normalize_vector(np.asarray(layer_stats["vision_error_proxy"], dtype=np.float64)), passes=3)
        dense_x = np.linspace(layers[0], layers[-1], 260)
        dense_text = smooth_curve(np.interp(dense_x, layers, text_err), passes=2)
        dense_vision = smooth_curve(np.interp(dense_x, layers, vision_err), passes=2)

        ax_curve.axvspan(-0.5, 12, color="#F2F3F0", zorder=0)
        ax_curve.axvspan(12, 24, color="#EEF5F5", zorder=0)
        ax_curve.axvspan(24, 35.5, color="#F7F4F1", zorder=0)
        # Stage labels are placed inside the plot, tight to the upper edge.
        ax_curve.text(6, 1.025, "Shallow\n(Perception)", fontsize=5.35, fontstyle="italic", color="#8A8F94", ha="center", va="top")
        ax_curve.text(18, 1.025, "Middle\n(Alignment)", fontsize=5.35, fontstyle="italic", color="#8A8F94", ha="center", va="top")
        ax_curve.text(29.5, 1.025, "Deep\n(Generation)", fontsize=5.35, fontstyle="italic", color="#8A8F94", ha="center", va="top")

        ax_curve.plot(dense_x, dense_text, color=mountain_blue, linewidth=1.75, label="Text", solid_capstyle="round")
        ax_curve.plot(dense_x, dense_vision, color=coral, linewidth=1.75, label="Vision", solid_capstyle="round")
        ax_curve.set_xlim(0, 35)
        ax_curve.set_ylim(0, 1.08)
        ax_curve.set_xticks([0, 12, 24, 35])
        ax_curve.set_yticks([0.0, 0.5, 1.0])
        ax_curve.set_xlabel("Decoder Layer", fontsize=6.8, labelpad=1.5)
        ax_curve.set_ylabel("Relative Quantization Error", fontsize=6.8, labelpad=1.5)
        ax_curve.grid(axis="y", color=grid_col, linewidth=0.5)
        leg = ax_curve.legend(loc="lower right", frameon=True, fontsize=6.25, handlelength=1.45, borderpad=0.28, labelspacing=0.25)
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#DCD9D2")
        frame.set_linewidth(0.5)
        frame.set_alpha(0.82)
        for spine in ["top", "right"]:
            ax_curve.spines[spine].set_visible(False)
        ax_curve.spines["left"].set_color("#CFD8DC")
        ax_curve.spines["bottom"].set_color("#BFBDB7")
        ax_curve.tick_params(axis="both", colors=axis_col, labelsize=6.35, length=2.0, pad=1.2)

    pdf = os.path.join(output_dir, f"{output_prefix}.pdf")
    png = os.path.join(output_dir, f"{output_prefix}.png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)
    return pdf, png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--layer-mode", default="middle", choices=["middle", "last", "mean"])
    parser.add_argument("--reuse-summary", action="store_true", help="redraw from existing summary without rerunning the model")
    parser.add_argument("--output-prefix", default="adamas_token_type_activation_graph", help="output filename prefix for pdf/png")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summary_path = os.path.join(args.output_dir, "adamas_token_type_activation_graph_summary.json")
    if args.reuse_summary and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        stats["values"] = np.asarray(stats.get("values", []), dtype=np.float64)
        stats["ablation_impact"] = np.asarray(stats.get("ablation_impact", []), dtype=np.float64)
        stats["ablation_signed_delta"] = np.asarray(stats.get("ablation_signed_delta", []), dtype=np.float64)
    else:
        stats = compute_stats(args.model_dir, args.output_dir, args.device, args.layer_mode)
    stats["layer_evolution"] = load_layer_evolution(args.output_dir)
    pdf, png = draw_figure(stats, args.output_dir, args.output_prefix)
    summary = dict(stats)
    summary["values"] = np.asarray(stats["values"]).tolist()
    summary["ablation_impact"] = np.asarray(stats["ablation_impact"]).tolist()
    summary["ablation_signed_delta"] = np.asarray(stats["ablation_signed_delta"]).tolist()
    with open(os.path.join(args.output_dir, "adamas_token_type_activation_graph_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
