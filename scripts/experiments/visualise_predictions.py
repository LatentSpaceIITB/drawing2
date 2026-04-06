"""
visualise_predictions.py
========================
Overlay predicted graph (boxes + edges) on a P&ID image alongside the GT graph.
Produces side-by-side PNG files for qualitative paper figures.

Output per image:
  <out_dir>/<stem>_pred.png   — image with predicted boxes + edges
  <out_dir>/<stem>_gt.png     — image with GT boxes + edges
  <out_dir>/<stem>_compare.png — side-by-side comparison

Usage:
  python scripts/experiments/visualise_predictions.py \\
      --checkpoint third_party/relationformer/trained_weights/pid/runs/E2_synth_only_fold0_v2_42/best_model.pth \\
      --experiment E2_synth_only \\
      --fold 0 \\
      --out-dir outputs/visualisations/E2_fold0

  # Visualise all 3 val images for a fold:
  python scripts/experiments/visualise_predictions.py \\
      --checkpoint <ckpt> --experiment E2_synth_only --fold 0 \\
      --out-dir outputs/visualisations/E2_fold0 --all-val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Repo / model path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RF_DIR    = REPO_ROOT / "third_party" / "relationformer"
sys.path.insert(0, str(RF_DIR))

from dataset_pid import build_pid_data_from_split, pid_collate_fn, NODE_CLASSES, PREDICATES
from models import build_model
from inference import graph_infer
from util.box_ops import box_cxcywh_to_xyxy
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Colour palette — one colour per node class
# ---------------------------------------------------------------------------
CLASS_COLOURS = {
    1: (220,  50,  50),   # valve        — red
    2: ( 50, 150, 220),   # pump         — blue
    3: ( 50, 200,  50),   # instrumentation — green
    4: (200, 150,  50),   # general      — orange
    5: (150,  50, 200),   # tank         — purple
    6: (200, 200,  50),   # arrow        — yellow
    7: ( 50, 200, 200),   # inlet/outlet — cyan
}
EDGE_COLOURS = {
    1: (255,  80,  80),   # solid        — bright red
    2: ( 80, 180, 255),   # non-solid    — bright blue
}
BG_COLOUR    = (30, 30, 30)
FONT_SIZE    = 14


def _try_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def denorm_boxes(boxes_cxcywh: np.ndarray, W: int, H: int) -> np.ndarray:
    """Convert normalised cxcywh [0,1] → absolute pixel xyxy."""
    xyxy = box_cxcywh_to_xyxy(torch.tensor(boxes_cxcywh)).numpy()
    xyxy[:, [0, 2]] *= W
    xyxy[:, [1, 3]] *= H
    return xyxy.astype(int)


def draw_graph(
    image: Image.Image,
    boxes_abs: np.ndarray,      # [N, 4] absolute xyxy
    labels: np.ndarray,         # [N] 0-indexed class ids
    edges: np.ndarray,          # [M, 3] node_i, node_j, pred_id  (1-indexed pred_id)
    title: str = "",
    score_thresh: float = 0.0,
    scores: np.ndarray | None = None,
) -> Image.Image:
    """Draw boxes and edges on a copy of the image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font  = _try_font(FONT_SIZE)
    font_sm = _try_font(max(FONT_SIZE - 4, 8))

    W, H = img.size

    # Draw edge lines first (under boxes)
    node_centres = []
    for i, box in enumerate(boxes_abs):
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        node_centres.append((cx, cy))

    for edge in edges:
        ni, nj, pid = int(edge[0]), int(edge[1]), int(edge[2])
        if ni >= len(node_centres) or nj >= len(node_centres):
            continue
        colour = EDGE_COLOURS.get(pid, (200, 200, 200))
        draw.line([node_centres[ni], node_centres[nj]],
                  fill=colour + (180,), width=2)

    # Draw boxes
    for i, (box, label) in enumerate(zip(boxes_abs, labels)):
        if scores is not None and scores[i] < score_thresh:
            continue
        cls_id  = int(label) + 1   # labels are 0-indexed, CLASS_COLOURS is 1-indexed
        colour  = CLASS_COLOURS.get(cls_id, (180, 180, 180))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Expand tiny boxes so they're visible
        pad = max(0, 4 - (x2 - x1))
        x1, x2 = x1 - pad, x2 + pad
        pad = max(0, 4 - (y2 - y1))
        y1, y2 = y1 - pad, y2 + pad

        draw.rectangle([x1, y1, x2, y2], outline=colour + (255,), width=2)
        cls_name = NODE_CLASSES[cls_id] if cls_id < len(NODE_CLASSES) else str(cls_id)
        abbrev   = cls_name[:3].upper()
        draw.text((x1 + 1, y1 + 1), abbrev, fill=(0, 0, 0, 255),   font=font_sm)
        draw.text((x1,     y1    ), abbrev, fill=colour + (255,), font=font_sm)

    # Title bar
    if title:
        draw.rectangle([0, 0, W, FONT_SIZE + 6], fill=(0, 0, 0, 200))
        draw.text((4, 3), title, fill=(255, 255, 255, 255), font=font)

    return img


def legend_strip(W: int) -> Image.Image:
    """Create a legend bar showing node class colours."""
    H = 30
    img   = Image.new("RGB", (W, H), color=(20, 20, 20))
    draw  = ImageDraw.Draw(img)
    font  = _try_font(11)
    x     = 4
    for cls_id, colour in CLASS_COLOURS.items():
        cls_name = NODE_CLASSES[cls_id] if cls_id < len(NODE_CLASSES) else str(cls_id)
        draw.rectangle([x, 6, x + 12, 22], fill=colour)
        draw.text((x + 14, 8), cls_name, fill=(220, 220, 220), font=font)
        x += 14 + len(cls_name) * 7 + 8
    for pred_id, colour in EDGE_COLOURS.items():
        pname = PREDICATES[pred_id] if pred_id < len(PREDICATES) else str(pred_id)
        draw.line([(x, 14), (x + 20, 14)], fill=colour, width=3)
        draw.text((x + 22, 8), pname, fill=(220, 220, 220), font=font)
        x += 22 + len(pname) * 7 + 8
    return img


# ---------------------------------------------------------------------------
# Config loader (same pattern as train_pid.py)
# ---------------------------------------------------------------------------

class _O:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _O(v) if isinstance(v, dict) else v)


def load_config(path: str):
    with open(path) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    return json.loads(json.dumps(raw), object_hook=lambda d: _O(d))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualise_one(
    model,
    config,
    img_tensor: torch.Tensor,
    target: dict,
    device: torch.device,
    out_dir: Path,
    stem: str,
    score_thresh: float = 0.3,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original image for display (full resolution)
    orig_img_path = target.get("image_path", None)
    if orig_img_path and Path(orig_img_path).exists():
        orig_img = Image.open(orig_img_path).convert("RGB")
    else:
        # Reconstruct from tensor (de-normalise)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        t = img_tensor.cpu().numpy().transpose(1, 2, 0)
        t = (t * std + mean).clip(0, 1)
        orig_img = Image.fromarray((t * 255).astype(np.uint8))

    W, H = orig_img.size

    # ---- GT graph ----
    gt_boxes_cxcywh = target["boxes"].float().numpy()
    gt_boxes_abs    = denorm_boxes(gt_boxes_cxcywh, W, H)
    gt_labels       = target["labels"].numpy() - 1   # 0-indexed
    gt_edges        = target["edges"].numpy()

    gt_img = draw_graph(
        orig_img, gt_boxes_abs, gt_labels, gt_edges,
        title=f"GT  |  {len(gt_boxes_abs)} nodes  {len(gt_edges)} edges"
    )

    # ---- Predicted graph ----
    with torch.no_grad():
        h, out = model([img_tensor.to(device)])
    relation_embed = model.relation_embed if not hasattr(model, "module") \
                     else model.module.relation_embed
    infer_out = graph_infer(h, out, relation_embed,
                            freq=None, emb=config.MODEL.DECODER.ADD_EMB_REL)

    pred_boxes_cxcywh = np.array(infer_out["pred_boxes"][0])     # [N, 4] cxcywh [0,1]
    pred_labels       = infer_out["pred_boxes_class"][0]          # [N] 0-indexed
    pred_scores       = infer_out["pred_boxes_score"][0]          # [N]
    pred_node_pairs   = infer_out["all_node_pairs"][0]            # [M, 2]
    pred_rel_scores   = infer_out["all_relation"][0]              # [M, 3]

    pred_boxes_abs = denorm_boxes(pred_boxes_cxcywh, W, H)

    # Build predicted edges: keep top-scoring relations above threshold
    pred_edges = []
    if pred_node_pairs is not None and pred_rel_scores is not None:
        pred_rel_class = 1 + pred_rel_scores[:, 1:].argmax(1)   # 1-indexed pred_id
        pred_rel_conf  = pred_rel_scores[:, 1:].max(1)
        for i, (pair, pid, conf) in enumerate(zip(pred_node_pairs, pred_rel_class, pred_rel_conf)):
            if conf > score_thresh:
                pred_edges.append([int(pair[0]), int(pair[1]), int(pid)])
    pred_edges_arr = np.array(pred_edges, dtype=int) if pred_edges \
                     else np.zeros((0, 3), dtype=int)

    n_above_thresh = (pred_scores >= score_thresh).sum()
    pred_img = draw_graph(
        orig_img,
        pred_boxes_abs,
        pred_labels.astype(int),
        pred_edges_arr,
        title=f"Pred (score≥{score_thresh})  |  {n_above_thresh} nodes  {len(pred_edges)} edges",
        score_thresh=score_thresh,
        scores=pred_scores,
    )

    # ---- Side-by-side ----
    leg   = legend_strip(W * 2)
    total_H = H + leg.height
    compare = Image.new("RGB", (W * 2, total_H), color=BG_COLOUR)
    compare.paste(gt_img,   (0, 0))
    compare.paste(pred_img, (W, 0))
    compare.paste(leg,      (0, H))

    gt_img.save(out_dir / f"{stem}_gt.png")
    pred_img.save(out_dir / f"{stem}_pred.png")
    compare.save(out_dir / f"{stem}_compare.png")
    print(f"  Saved → {out_dir / stem}_compare.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise P&ID predictions vs GT")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--experiment", default="E2_synth_only")
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--out-dir",    type=Path,
                        default=REPO_ROOT / "outputs" / "visualisations")
    parser.add_argument("--score-thresh", type=float, default=0.3,
                        help="Min prediction score to include in visualisation")
    parser.add_argument("--n-images",   type=int, default=3,
                        help="Number of val images to visualise (default: all val)")
    parser.add_argument("--device",     default="cuda:0")
    args = parser.parse_args()

    config = load_config(str(RF_DIR / "configs" / "pid.yaml"))

    # Resolve data paths
    for attr in ("PID_REAL_COCO_JSON", "PID_SYNTH_COCO_JSON", "PID_SPLITS_JSON"):
        val = getattr(config.DATA, attr)
        if val and not Path(val).is_absolute():
            setattr(config.DATA, attr, str(REPO_ROOT / val))

    _, val_ds = build_pid_data_from_split(
        config,
        splits_json     = config.DATA.PID_SPLITS_JSON,
        fold            = args.fold,
        experiment      = args.experiment,
        synth_coco_json = config.DATA.PID_SYNTH_COCO_JSON,
        real_coco_json  = config.DATA.PID_REAL_COCO_JSON,
    )

    # Inject image paths into dataset items for display
    # (dataset __getitem__ doesn't return the path, so we patch it)
    orig_getitem = val_ds.__getitem__
    def getitem_with_path(idx):
        img, tgt = orig_getitem(idx)
        tgt["image_path"] = val_ds.data_dicts[idx]["image"]
        return img, tgt
    val_ds.__getitem__ = getitem_with_path

    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=pid_collate_fn, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = build_model(config).to(device)
    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    state  = ckpt.get("net", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    out_dir = args.out_dir / f"{args.experiment}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualisations → {out_dir}")

    for i, (imgs, targets) in enumerate(loader):
        if i >= args.n_images:
            break
        stem = Path(targets[0]["image_path"]).stem
        print(f"  Image {i+1}: {stem}")
        visualise_one(
            model, config,
            imgs[0], targets[0],
            device, out_dir, stem,
            score_thresh=args.score_thresh,
        )

    print(f"\nDone. {min(i+1, args.n_images)} images written to {out_dir}")


if __name__ == "__main__":
    main()
