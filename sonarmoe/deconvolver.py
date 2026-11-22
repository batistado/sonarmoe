"""Matched-Filter + MLP Timing Deconvolver, Fold Variance & Permutation-Calibrated MI."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SonarDeconvolverMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        hid = max(64, min(128, num_classes * 6))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hid, hid // 2),
            nn.GELU(),
            nn.Linear(hid // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _plugin_mi_bits(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    n = float(len(y_true))
    conf = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(conf, (y_true, y_pred), 1.0)
    joint = conf / n
    p_y = joint.sum(axis=1, keepdims=True)
    p_yhat = joint.sum(axis=0, keepdims=True)
    mi = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            if joint[i, j] > 0.0 and p_y[i, 0] > 0.0 and p_yhat[0, j] > 0.0:
                mi += joint[i, j] * math.log2(joint[i, j] / (p_y[i, 0] * p_yhat[0, j]))
    return mi


def permutation_calibrated_mi_bits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    num_permutations: int = 200,
    seed: int = 20261118,
) -> tuple[float, float]:
    """Computes permutation-debiased discrete MI I_perm(Y; Y_hat) in bits.

    Eliminates finite-sample upward bias even when K^2 > N (e.g., K=20, N=160),
    where asymptotic Miller-Madow correction under-compensates sparse contingency tables.
    Returns (calibrated_mi_bits, raw_plugin_mi_bits).
    """
    raw_mi = _plugin_mi_bits(y_true, y_pred, num_classes)
    rng = np.random.default_rng(seed)
    null_vals = np.empty(num_permutations, dtype=np.float64)
    for b in range(num_permutations):
        y_perm = rng.permutation(y_true)
        null_vals[b] = _plugin_mi_bits(y_perm, y_pred, num_classes)
    null_mean = float(null_vals.mean())
    return max(0.0, raw_mi - null_mean), raw_mi


def evaluate_deconvolver_kfold(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    num_folds: int = 5,
    seed: int = 20261118,
) -> dict[str, Any]:
    """Blind stratified K-fold evaluation returning Top-1, Top-3, Macro-F1, fold std, and MI."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_samples, _ = features.shape
    num_classes = len(class_names)

    row_centered = features - features.mean(axis=1, keepdims=True)
    aug_features = np.concatenate([features, row_centered], axis=1)
    in_dim = aug_features.shape[1]

    fold_ids = np.zeros(num_samples, dtype=np.int64)
    for c in range(num_classes):
        c_indices = np.where(labels == c)[0]
        for idx_pos, sample_idx in enumerate(c_indices):
            fold_ids[sample_idx] = idx_pos % num_folds

    oof_preds = np.zeros(num_samples, dtype=np.int64)
    oof_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
    fold_top1_accs: list[float] = []
    fold_top3_accs: list[float] = []

    for fold in range(num_folds):
        val_mask = fold_ids == fold
        tr_mask = ~val_mask

        x_tr = aug_features[tr_mask].astype(np.float32)
        y_tr = labels[tr_mask]
        x_val = aug_features[val_mask].astype(np.float32)
        y_val = labels[val_mask]

        mu = x_tr.mean(axis=0, keepdims=True)
        sd = x_tr.std(axis=0, keepdims=True) + 1e-5
        x_tr_n = (x_tr - mu) / sd
        x_val_n = (x_val - mu) / sd

        xt = torch.from_numpy(x_tr_n)
        yt = torch.from_numpy(y_tr)

        lin = nn.Linear(in_dim, num_classes)
        opt_l = torch.optim.AdamW(lin.parameters(), lr=1.5e-2, weight_decay=0.20)
        for _ in range(160):
            opt_l.zero_grad()
            F.cross_entropy(lin(xt), yt).backward()
            opt_l.step()

        clf = SonarDeconvolverMLP(in_dim, num_classes)
        opt_m = torch.optim.AdamW(clf.parameters(), lr=6e-3, weight_decay=4e-2)
        clf.train()
        for _ in range(160):
            opt_m.zero_grad()
            F.cross_entropy(clf(xt), yt).backward()
            opt_m.step()

        clf.eval()
        lin.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val_n)
            p_lin = F.softmax(lin(xv), dim=-1).numpy()
            p_mlp = F.softmax(clf(xv), dim=-1).numpy()
            probs = 0.55 * p_lin + 0.45 * p_mlp

        preds_f = probs.argmax(axis=-1)
        oof_probs[val_mask] = probs
        oof_preds[val_mask] = preds_f
        fold_top1_accs.append(float((preds_f == y_val).mean()) * 100.0)
        k_top = min(3, num_classes)
        t3_f = np.argsort(-probs, axis=-1)[:, :k_top]
        fold_top3_accs.append(float(np.any(t3_f == y_val[:, None], axis=-1).mean()) * 100.0)

    top1_acc = float((oof_preds == labels).mean())
    k_top = min(3, num_classes)
    top3_idx = np.argsort(-oof_probs, axis=-1)[:, :k_top]
    top3_acc = float(np.any(top3_idx == labels[:, None], axis=-1).mean())

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int32)
    for yt_i, yp_i in zip(labels, oof_preds):
        conf_mat[yt_i, yp_i] += 1

    f1_per_class: dict[str, float] = {}
    f1_vals: list[float] = []
    for c in range(num_classes):
        tp = float(conf_mat[c, c])
        fp = float(conf_mat[:, c].sum() - tp)
        fn = float(conf_mat[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_vals.append(f1)
        f1_per_class[class_names[c]] = round(f1 * 100.0, 2)

    macro_f1 = float(np.mean(f1_vals))

    mi_perm, mi_raw = permutation_calibrated_mi_bits(
        labels, oof_preds, num_classes, num_permutations=200, seed=seed
    )
    eps_prob = np.clip(oof_probs[np.arange(num_samples), labels], 1e-6, 1.0)
    ce_bits = float(-np.mean(np.log2(eps_prob)))
    mi_variational = max(0.0, math.log2(num_classes) - ce_bits)

    return {
        "accuracy_pct": round(top1_acc * 100.0, 2),
        "accuracy_fold_std_pct": round(float(np.std(fold_top1_accs, ddof=1)), 2),
        "top3_accuracy_pct": round(top3_acc * 100.0, 2),
        "top3_fold_std_pct": round(float(np.std(fold_top3_accs, ddof=1)), 2),
        "macro_f1_pct": round(macro_f1 * 100.0, 2),
        "empirical_mi_bits": round(mi_perm, 4),
        "raw_plugin_mi_bits": round(mi_raw, 4),
        "variational_mi_bits": round(mi_variational, 4),
        "per_class_f1_pct": f1_per_class,
        "confusion_matrix": conf_mat.tolist(),
    }
