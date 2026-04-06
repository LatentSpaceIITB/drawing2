"""
verify_eval_pipeline.py
=======================
Pre-flight check: run before every new training configuration to confirm
the evaluation pipeline is working correctly.

Checks:
  1. Box scale ratio (train synth / val real) is < 3x
  2. Perfect predictions score R@100 > 0.5
  3. Loss computation works (no None losses)
  4. Single DDP step runs without crash (single GPU)

Exits with code 0 if all checks pass, 1 if any fail.

Usage:
  python scripts/experiments/verify_eval_pipeline.py
  python scripts/experiments/verify_eval_pipeline.py --fold 1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RF_DIR    = REPO_ROOT / "third_party" / "relationformer"
sys.path.insert(0, str(RF_DIR))

from dataset_pid import build_pid_data_from_split, pid_collate_fn
from models import build_model
from models.matcher_scene import build_matcher
from losses import SetCriterion
from inference import graph_infer
from util.box_ops import box_cxcywh_to_xyxy
from util.sg_recall import BasicSceneGraphEvaluator
from torch.utils.data import DataLoader


class _O:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _O(v) if isinstance(v, dict) else v)


def load_config():
    with open(RF_DIR / "configs" / "pid.yaml") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    config = json.loads(json.dumps(raw), object_hook=lambda d: _O(d))
    # Resolve relative paths
    for attr in ("PID_REAL_COCO_JSON", "PID_SYNTH_COCO_JSON", "PID_SPLITS_JSON"):
        val = getattr(config.DATA, attr)
        if val and not Path(val).is_absolute():
            setattr(config.DATA, attr, str(REPO_ROOT / val))
    return config


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    return condition


def main(fold: int = 0) -> int:
    print(f"\n{'='*60}")
    print(f"  Eval pipeline verification  (fold={fold})")
    print(f"{'='*60}")
    config = load_config()
    all_pass = True

    # ---- Build datasets ----
    train_ds, val_ds = build_pid_data_from_split(
        config,
        splits_json     = config.DATA.PID_SPLITS_JSON,
        fold            = fold,
        experiment      = "E2_synth_only",
        synth_coco_json = config.DATA.PID_SYNTH_COCO_JSON,
        real_coco_json  = config.DATA.PID_REAL_COCO_JSON,
    )
    t_loader = DataLoader(train_ds, batch_size=1, shuffle=False,
                          collate_fn=pid_collate_fn, num_workers=0)
    v_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                          collate_fn=pid_collate_fn, num_workers=0)
    print(f"\n  Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    # ---- Check 1: box scale ratio ----
    print("\n[1] Box scale ratio (train synth vs val real)")
    t_imgs, t_tgts = next(iter(t_loader))
    v_imgs, v_tgts = next(iter(v_loader))
    t_w = (box_cxcywh_to_xyxy(t_tgts[0]["boxes"].float()).numpy()[:, 2] -
           box_cxcywh_to_xyxy(t_tgts[0]["boxes"].float()).numpy()[:, 0])
    v_w = (box_cxcywh_to_xyxy(v_tgts[0]["boxes"].float()).numpy()[:, 2] -
           box_cxcywh_to_xyxy(v_tgts[0]["boxes"].float()).numpy()[:, 0])
    ratio = float(np.median(t_w) / max(np.median(v_w), 1e-8))
    ok = check("Scale ratio < 3x", ratio < 3.0,
               f"synth_median={np.median(t_w):.5f}  real_median={np.median(v_w):.5f}  ratio={ratio:.2f}x")
    all_pass = all_pass and ok

    # ---- Check 2: perfect-prediction R@K ----
    print("\n[2] Perfect-prediction R@100 > 0.5")
    sg = BasicSceneGraphEvaluator.all_modes(multiple_preds=False, config=config)
    sg['sgdet'].reset()
    for imgs, tgts in v_loader:
        gt = tgts[0]
        gt_cls_0idx = gt["labels"].numpy() - 1
        gt_edges    = gt["edges"].numpy()
        gt_entry = {"boxes": gt["boxes"].float(), "labels": gt_cls_0idx, "edges": gt_edges}
        edge_scores = np.zeros((len(gt_edges), 3))
        for i, e in enumerate(gt_edges):
            edge_scores[i, int(e[2])] = 1.0
        pred_cls  = {"labels": gt_cls_0idx.astype(float),
                     "scores": np.ones(len(gt_cls_0idx)),
                     "boxes":  gt["boxes"].float()}
        pred_edge = {"node_pair": gt_edges[:, :2], "edge_score": edge_scores}
        sg['sgdet'].evaluate_scene_graph_entry(gt_entry, [pred_cls, pred_edge])
    buf  = sg['sgdet']._buffers
    r100 = float(np.mean(buf[100])) if buf[100] else 0.0
    ok   = check("Perfect-pred R@100 > 0.5", r100 > 0.5, f"R@100={r100:.4f}")
    all_pass = all_pass and ok

    # ---- Check 3: loss computation ----
    print("\n[3] Loss computation (no None values)")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = build_model(config).to(device)
    model.train()
    matcher   = build_matcher(config=config)
    criterion = SetCriterion(config, matcher, model.relation_embed,
                             freq_baseline=None, use_target=True, focal_alpha=None).to(device)
    imgs, tgts = next(iter(t_loader))
    tgt = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
           for t in tgts]
    h, out = model([imgs[0].to(device)])
    loss_dict = criterion(h, out, tgt)
    none_losses = [k for k, v in loss_dict.items() if v is None]
    ok = check("No None losses", len(none_losses) == 0,
               f"None losses: {none_losses}" if none_losses else
               f"total={loss_dict['total'].item():.4f}")
    all_pass = all_pass and ok

    # ---- Check 4: backward pass ----
    print("\n[4] Backward pass (no crash)")
    try:
        loss_dict["total"].backward()
        ok = check("Backward pass", True, "OK")
    except Exception as e:
        ok = check("Backward pass", False, str(e))
    all_pass = all_pass and ok

    # ---- Summary ----
    print(f"\n{'='*60}")
    if all_pass:
        print("  ALL CHECKS PASSED — safe to start training")
    else:
        print("  SOME CHECKS FAILED — fix issues before training")
    print(f"{'='*60}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()
    sys.exit(main(args.fold))
