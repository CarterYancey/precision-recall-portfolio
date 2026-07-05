from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    requested_recall: float
    requested_precision: float
    achieved_recall: float
    achieved_precision: float
    tp: int
    fp: int
    fn: int
    tn: int
    selected_names: List[str]
    num_ways: int
    note: Optional[str] = None


def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def _validate_metric(x: float, name: str) -> None:
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}.")
    if x < 0.0 or x > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x}.")


def estimate_num_ways(
    df: pd.DataFrame,
    name_col: str,
    label_col: str,
    recall: float,
    precision: float,
    *,
    positive_label: str = "TRUE",
) -> int:
    """
    Estimate the number of distinct subsets of names that satisfy the requested (recall, precision),
    under the same rounding scheme used by simulate_selection:

      TP = round(recall * T)

    If precision > 0:
      FP = round(TP * (1 - precision) / precision)
      ways = C(T, TP) * C(F, FP)  if feasible else 0

    Special cases:
      - If TP==0 and precision==1: only empty selection achieves precision=1 -> 1 way
      - If TP==0 and precision==0: any non-empty selection of negatives achieves precision=0
            ways = 2^F - 1
      - If TP>0 and precision==0: impossible -> 0
      - If T==0:
          * recall>0 -> 0
          * recall==0, precision==1 -> 1
          * recall==0, precision==0 -> 2^F - 1
          * recall==0, precision in (0,1) -> 0
    """
    _validate_metric(recall, "recall")
    _validate_metric(precision, "precision")

    labels = df[label_col].astype(str)
    pos_mask = labels.eq(str(positive_label))
    T = int(pos_mask.sum())
    F = int((~pos_mask).sum())

    # No positives in dataset
    if T == 0:
        if recall > 0:
            return 0
        if _is_close(precision, 1.0):
            return 1
        if _is_close(precision, 0.0):
            return (1 << F) - 1  # 2^F - 1
        return 0

    tp = int(round(recall * T))
    tp = max(0, min(tp, T))

    if _is_close(precision, 0.0):
        return 0 if tp > 0 else ((1 << F) - 1)  # any non-empty neg subset

    if _is_close(precision, 1.0) and tp == 0:
        return 1  # must select nothing

    # precision in (0,1] with tp>=0
    fp = int(round(tp * (1.0 - precision) / precision))
    if fp < 0 or fp > F:
        return 0

    # If tp==0, fp will be 0; that yields precision=1, so any request in (0,1) is infeasible
    if tp == 0 and not _is_close(precision, 1.0):
        return 0

    return math.comb(T, tp) * math.comb(F, fp)


def simulate_selection(
    df: pd.DataFrame,
    name_col: str,
    label_col: str,
    recall: float,
    precision: float,
    *,
    positive_label: str = "TRUE",
    random_state: Optional[int] = None,
    strict: bool = False,
) -> SimulationResult:
    """
    Simulate a "model-selected" list of names from df that targets (recall, precision).

    Construction:
      TP = round(recall * T)
      FP = round(TP * (1 - precision) / precision)  (when precision > 0)
      Select TP positives and FP negatives uniformly at random.

    If strict=True, raises ValueError on infeasible requests.
    Otherwise, returns best-effort with a note and reports achieved metrics.
    """
    _validate_metric(recall, "recall")
    _validate_metric(precision, "precision")

    # Split positives/negatives
    labels = df[label_col].astype(str)
    pos_mask = labels.eq(str(positive_label))
    pos_df = df.loc[pos_mask, [name_col]]
    neg_df = df.loc[~pos_mask, [name_col]]

    T = int(pos_df.shape[0])
    F = int(neg_df.shape[0])

    # Helper to compute achieved metrics safely
    def compute_metrics(tp_: int, fp_: int) -> Tuple[float, float]:
        achieved_recall = (tp_ / T) if T > 0 else (1.0 if tp_ == 0 else 0.0)
        denom = tp_ + fp_
        achieved_precision = (tp_ / denom) if denom > 0 else 1.0
        return achieved_recall, achieved_precision

    note: Optional[str] = None

    # Precompute "ways" for the exact request (0 if infeasible under rounding scheme)
    ways_requested = estimate_num_ways(
        df, name_col, label_col, recall, precision, positive_label=positive_label
    )

    # If strict and infeasible, fail early.
    if strict and ways_requested == 0:
        raise ValueError("Infeasible (recall, precision) request for this dataset under the rounding scheme.")

    # Case: no positives
    if T == 0:
        if recall > 0:
            note = "Dataset has 0 TRUE items; any recall > 0 is impossible. Returning empty selection."
            achieved_recall, achieved_precision = compute_metrics(tp_=0, fp_=0)
            return SimulationResult(
                recall, precision, achieved_recall, achieved_precision,
                tp=0, fp=0, fn=0, tn=F,
                selected_names=[],
                num_ways=0,
                note=note,
            )

        if _is_close(precision, 1.0):
            achieved_recall, achieved_precision = compute_metrics(tp_=0, fp_=0)
            return SimulationResult(
                recall, precision, achieved_recall, achieved_precision,
                tp=0, fp=0, fn=0, tn=F,
                selected_names=[],
                num_ways=1,
                note="No TRUE items; selected nothing so precision is defined as 1.0.",
            )

        if _is_close(precision, 0.0):
            if F == 0:
                achieved_recall, achieved_precision = compute_metrics(tp_=0, fp_=0)
                return SimulationResult(
                    recall, precision, achieved_recall, achieved_precision,
                    tp=0, fp=0, fn=0, tn=0,
                    selected_names=[],
                    num_ways=0,
                    note="Dataset is empty; cannot achieve precision=0.",
                )
            # choose 1 negative (one valid outcome among many)
            selected = neg_df[name_col].sample(n=1, random_state=random_state).tolist()
            achieved_recall, achieved_precision = compute_metrics(tp_=0, fp_=1)
            return SimulationResult(
                recall, precision, achieved_recall, achieved_precision,
                tp=0, fp=1, fn=0, tn=F - 1,
                selected_names=selected,
                num_ways=(1 << F) - 1,
                note="No TRUE items; selected 1 FALSE item so precision is 0.0 by definition.",
            )

        achieved_recall, achieved_precision = compute_metrics(tp_=0, fp_=0)
        return SimulationResult(
            recall, precision, achieved_recall, achieved_precision,
            tp=0, fp=0, fn=0, tn=F,
            selected_names=[],
            num_ways=0,
            note="No TRUE items; requested precision in (0,1) is infeasible. Returning empty selection.",
        )

    # General case: T > 0
    tp = int(round(recall * T))
    tp = max(0, min(tp, T))

    # Precision=0 special handling
    if _is_close(precision, 0.0):
        if tp > 0:
            note = "Requested precision=0 with recall>0 is impossible. Returning recall-matching selection with FP=0."
            fp = 0
        else:
            fp = 1 if F > 0 else 0
            if F == 0:
                note = "No FALSE items available; cannot achieve precision=0. Returning empty selection."

        selected_true = pos_df[name_col].sample(n=tp, random_state=random_state).tolist() if tp else []
        selected_false = neg_df[name_col].sample(n=fp, random_state=random_state).tolist() if fp else []
        selected = selected_true + selected_false
        achieved_recall, achieved_precision = compute_metrics(tp_=tp, fp_=fp)

        num_ways = 0 if tp > 0 else ((1 << F) - 1 if F > 0 else 0)
        return SimulationResult(
            recall, precision, achieved_recall, achieved_precision,
            tp=tp, fp=fp, fn=T - tp, tn=F - fp,
            selected_names=selected,
            num_ways=num_ways,
            note=note,
        )

    # precision in (0, 1]
    fp = int(round(tp * (1.0 - precision) / precision))
    fp = max(0, fp)

    if fp > F:
        note = (f"Infeasible request: need FP={fp} but only F={F} FALSE exist. "
                f"Capping FP to {F} (achieved precision will be higher).")
        fp = F

    if tp == 0 and not _is_close(precision, 1.0):
        note = ("With TP=0, achievable precision is 1.0 (select none) or 0.0 (select any). "
                "Requested precision in (0,1) is infeasible; returning empty selection.")
        fp = 0

    selected_true = pos_df[name_col].sample(n=tp, random_state=random_state).tolist() if tp else []
    selected_false = neg_df[name_col].sample(n=fp, random_state=random_state).tolist() if fp else []
    selected = selected_true + selected_false

    achieved_recall, achieved_precision = compute_metrics(tp_=tp, fp_=fp)

    # If we had to cap/adjust, the number of ways is for the *returned* (tp, fp).
    num_ways_returned = math.comb(T, tp) * math.comb(F, fp) if (0 <= fp <= F) else 0

    return SimulationResult(
        recall, precision, achieved_recall, achieved_precision,
        tp=tp, fp=fp, fn=T - tp, tn=F - fp,
        selected_names=selected,
        num_ways=(ways_requested if note is None else num_ways_returned),
        note=note,
    )


# ---- Example usage ----
if __name__ == "__main__":
    df = pd.DataFrame(
        {"name": [f"n{i}" for i in range(500)],
         "label": ["TRUE"] * 150 + ["FALSE"] * 350}
    )

    #for r, p in [(1.0, 1.0), (0.5, 0.5), (0.1, 0.8), (0.2, 0.0)]:
    for r, p in [(0.1, 0.85), (0.85, 0.1)]:
        res = simulate_selection(df, "name", "label", r, p, random_state=69, strict=False)
        print(f"\nRequested (R,P)=({r},{p})")
        print(f"Achieved  (R,P)=({res.achieved_recall:.3f},{res.achieved_precision:.3f})")
        print(f"Selected: {res.selected_names}")
        print(f"TP={res.tp}, FP={res.fp}, #ways={res.num_ways}")
        if res.note:
            print("Note:", res.note)

