"""Scoring, following the parent project's conventions rather than inventing new ones.

Two of these are not the obvious choice and both come from AutoSegCT's own practice:

* **Lesions are scored per matched component**, each ground-truth lesion against the
  predicted component that overlaps it best. Aggregating every lesion into one mask and
  taking a single Dice understated the score by roughly 3x upstream and nearly derailed a
  real investigation, because two small objects a few millimetres apart score as one
  badly-placed blob.
* **Region agreement is reported as symmetric difference in millilitres** alongside Dice.
  Dice on a large structure is dominated by its interior and barely moves when a boundary
  is wrong by a millimetre, which on a phantom is the error that matters.

A structure absent from the ground truth is reported as absent, never as Dice 0. Scoring
a prediction against a structure that was never painted produces a number that looks like
a failure and means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import labels as L


@dataclass
class StructureScore:
    label: int
    name: str
    dice: float | None
    pred_ml: float
    true_ml: float
    sym_diff_ml: float | None

    def line(self) -> str:
        if self.dice is None:
            return f"  {self.name:9s} absent from GT (predicted {self.pred_ml:.2f} mL)"
        return (f"  {self.name:9s} dice {self.dice:.3f}   "
                f"pred {self.pred_ml:7.2f} mL   gt {self.true_ml:7.2f} mL   "
                f"symdiff {self.sym_diff_ml:7.2f} mL")


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = float(np.count_nonzero(pred & truth))
    total = float(np.count_nonzero(pred) + np.count_nonzero(truth))
    return 1.0 if total == 0 else 2.0 * inter / total


def score_structures(pred: np.ndarray, truth: np.ndarray,
                     ml_per_voxel: float) -> list[StructureScore]:
    """Per-structure Dice and symmetric difference over the whole class space."""
    out = []
    for label in L.STRUCTURES:
        p, t = pred == label, truth == label
        true_ml = float(t.sum()) * ml_per_voxel
        pred_ml = float(p.sum()) * ml_per_voxel
        if true_ml < L.NEGLIGIBLE_ML:
            out.append(StructureScore(label, L.CLASS_NAMES[label], None,
                                      pred_ml, true_ml, None))
            continue
        sym = float(np.count_nonzero(p ^ t)) * ml_per_voxel
        out.append(StructureScore(label, L.CLASS_NAMES[label], dice(p, t),
                                  pred_ml, true_ml, sym))
    return out


def lesion_component_scores(pred: np.ndarray, truth: np.ndarray,
                            ml_per_voxel: float) -> list[dict]:
    """Dice for each ground-truth lesion against its best-overlap predicted component.

    A ground-truth lesion with no overlapping prediction scores 0 and is reported as a
    miss rather than dropped, so the list length always equals the number of painted
    lesions.
    """
    t_lab, t_n = ndimage.label(truth == L.LESION)
    p_lab, p_n = ndimage.label(pred == L.LESION)
    results = []
    for i in range(1, t_n + 1):
        t_mask = t_lab == i
        vol = float(t_mask.sum()) * ml_per_voxel
        if p_n == 0:
            results.append({"gt_lesion": i, "gt_ml": vol, "dice": 0.0, "matched": None})
            continue
        overlaps = np.bincount(p_lab[t_mask], minlength=p_n + 1)
        overlaps[0] = 0
        best = int(overlaps.argmax())
        if overlaps[best] == 0:
            results.append({"gt_lesion": i, "gt_ml": vol, "dice": 0.0, "matched": None})
            continue
        results.append({"gt_lesion": i, "gt_ml": vol,
                        "dice": dice(p_lab == best, t_mask), "matched": best})
    return results


def degenerate_baselines(truth: np.ndarray, covered: np.ndarray,
                         ml_per_voxel: float, seed: int = 0) -> str:
    """Where the floor actually is, for each structure.

    Without this the claim "the lobe sits at the floor" rests on an assumption. The lobe
    is the largest structure by far, so a classifier that simply predicts lobe everywhere
    should score a *high* lobe Dice. If it does, then a low lobe score is well below
    trivial and means something is wrong with the lobe arm rather than the lobe being
    unreachable.
    """
    rng = np.random.default_rng(seed)
    inside = truth[covered]
    prior = np.bincount(inside, minlength=len(L.CLASS_NAMES)).astype(float)
    prior /= prior.sum()

    arms: dict[str, np.ndarray] = {"all background": np.zeros_like(truth)}
    for label in L.STRUCTURES:
        arm = np.zeros_like(truth)
        arm[covered] = label
        arms[f"all {L.CLASS_NAMES[label]}"] = arm
    drawn = np.zeros_like(truth)
    drawn[covered] = rng.choice(len(prior), size=int(covered.sum()), p=prior)
    arms["random (class prior)"] = drawn

    lines = ["degenerate baselines (where the floor is):"]
    header = "  " + " " * 22 + "".join(f"{L.CLASS_NAMES[c][:8]:>10s}"
                                       for c in L.STRUCTURES)
    lines.append(header)
    for name, arm in arms.items():
        cells = []
        for label in L.STRUCTURES:
            t = truth == label
            if float(t.sum()) * ml_per_voxel < L.NEGLIGIBLE_ML:
                cells.append(f"{'--':>10s}")
            else:
                cells.append(f"{dice(arm == label, t):>10.3f}")
        lines.append(f"  {name:22s}" + "".join(cells))
    return "\n".join(lines)


def lesion_confusion(pred: np.ndarray, truth: np.ndarray,
                     ml_per_voxel: float) -> str:
    """Where ground-truth lesion voxels actually go.

    Lesions read 94/364/674 HU against a capsule at 190/710/1508 — heavily overlapping,
    and in patch-mean contrast both are positive blobs. So the interesting failure is not
    "lesions were missed" but "lesions were absorbed into which class", and a Dice of
    zero cannot tell those apart.
    """
    mask = truth == L.LESION
    total = int(mask.sum())
    if total == 0:
        return "lesion confusion: no lesion voxels in this region"
    counts = np.bincount(pred[mask], minlength=len(L.CLASS_NAMES))
    parts = [f"{L.CLASS_NAMES[c]} {100.0 * counts[c] / total:.1f}%"
             for c in range(len(counts)) if counts[c]]
    return (f"lesion confusion ({total * ml_per_voxel:.2f} mL of GT lesion) -> "
            + ", ".join(parts))


def report(name: str, pred: np.ndarray, truth: np.ndarray,
           ml_per_voxel: float) -> str:
    """Human-readable score block for one prediction."""
    lines = [f"{name}:"]
    for score in score_structures(pred, truth, ml_per_voxel):
        lines.append(score.line())
    lesions = lesion_component_scores(pred, truth, ml_per_voxel)
    if lesions:
        lines.append("  lesions, per matched component:")
        for item in lesions:
            match = "MISS" if item["matched"] is None else f"comp {item['matched']}"
            lines.append(f"    gt lesion {item['gt_lesion']} "
                         f"({item['gt_ml']:.2f} mL): dice {item['dice']:.3f}  {match}")
    return "\n".join(lines)
