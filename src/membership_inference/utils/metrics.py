from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.metrics import recall_score, roc_auc_score, roc_curve


def compute_binary_mia_metrics(
    y: npt.NDArray,
    probs: npt.NDArray,
    preds: Optional[npt.NDArray] = None,
) -> dict[str, float]:
    """Compute standard membership inference evaluation metrics."""
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    preds = np.asarray(preds) if preds is not None else (probs >= 0.5).astype(int)

    uar = recall_score(y, preds, average="macro")
    auc = roc_auc_score(y, probs)

    fpr, tpr, _ = roc_curve(y, probs, drop_intermediate=False)

    if np.all(np.isnan(fpr)) or np.all(np.isnan(tpr)):
        return {
            "uar": uar,
            "auc": auc,
            "eer": float(np.nan),
            "fpr@fpr=10%": float(np.nan),
            "tpr@fpr=10%": float(np.nan),
            "fpr@fpr=1%": float(np.nan),
            "tpr@fpr=1%": float(np.nan),
            "fpr@fpr=0.1%": float(np.nan),
            "tpr@fpr=0.1%": float(np.nan),
            "fpr@maxfpr=10%": float(np.nan),
            "tpr@maxfpr=10%": float(np.nan),
            "fpr@maxfpr=1%": float(np.nan),
            "tpr@maxfpr=1%": float(np.nan),
            "fpr@maxfpr=0.1%": float(np.nan),
            "tpr@maxfpr=0.1%": float(np.nan),
        }

    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = float((fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2)

    def fpr_and_tpr_at_target(target_fpr: float) -> tuple[float, float]:
        idxs = np.where(fpr >= target_fpr)[0]
        if len(idxs) == 0:
            return (float(np.nan), float(np.nan))
        idx = idxs[0]
        return (float(fpr[idx]), float(tpr[idx]))

    def fpr_and_tpr_at_budget(target_fpr: float) -> tuple[float, float]:
        idxs = np.where(fpr <= target_fpr)[0]
        if len(idxs) == 0:
            return (0.0, 0.0)
        idx = idxs[-1]
        return (float(fpr[idx]), float(tpr[idx]))

    fpr_at_10, tpr_at_10 = fpr_and_tpr_at_target(0.10)
    fpr_at_1, tpr_at_1 = fpr_and_tpr_at_target(0.01)
    fpr_at_01, tpr_at_01 = fpr_and_tpr_at_target(0.001)
    bfpr_at_10, btpr_at_10 = fpr_and_tpr_at_budget(0.10)
    bfpr_at_1, btpr_at_1 = fpr_and_tpr_at_budget(0.01)
    bfpr_at_01, btpr_at_01 = fpr_and_tpr_at_budget(0.001)

    return {
        "uar": uar,
        "auc": auc,
        "eer": eer,
        "fpr@fpr=10%": fpr_at_10,
        "tpr@fpr=10%": tpr_at_10,
        "fpr@fpr=1%": fpr_at_1,
        "tpr@fpr=1%": tpr_at_1,
        "fpr@fpr=0.1%": fpr_at_01,
        "tpr@fpr=0.1%": tpr_at_01,
        "fpr@maxfpr=10%": bfpr_at_10,
        "tpr@maxfpr=10%": btpr_at_10,
        "fpr@maxfpr=1%": bfpr_at_1,
        "tpr@maxfpr=1%": btpr_at_1,
        "fpr@maxfpr=0.1%": bfpr_at_01,
        "tpr@maxfpr=0.1%": btpr_at_01,
    }
