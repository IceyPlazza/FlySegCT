"""Loading, cropping, normalising and patch-sampling the phantom CT volumes.

Two facts from the parent AutoSegCT project shape everything here.

First, there is no single absolute intensity window that works. The same physical
phantom images at about +507 HU above background on the Xoran scanner and about +50 HU
on VUIIS, so the low-contrast stages upstream use a per-slice local z-score rather than
an HU threshold. This module produces both channels -- scaled HU and local z -- and lets
the caller decide, because for Xoran high-contrast scans the absolute HU genuinely is
informative and throwing it away would be a handicap the pipeline does not accept.

Second, the volumes are large: 732 x 732 x 420 at 0.3 mm isotropic, about 225 M voxels,
while the anatomy occupies only 5-7% of that and fits in a ~113 mm cube. Everything is
cropped to a scan-derived region of interest before any feature is computed. The ROI is
derived from the image, never from the ground truth, so no label information leaks into
the region the classifier is asked to work in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from . import labels as L

CORPUS = Path(r"C:\Users\iven0\OneDrive\Desktop\BPH_collections\organized_data")
MANIFEST = CORPUS / "manifest.csv"
ARM_DIR = {"hi": "hi_contrast", "lo": "lo_contrast"}

#: Documented out-of-corpus cases. `BPH_2` has its urethra against the scan edge;
#: `bph4_allcap` and `5_BPH` are tilted; the rest are superseded.
OUT_OF_SCOPE = frozenset({
    "BPH_2", "bph4_allcap", "5_BPH", "1_BPH_11_17_25", "BPH_3", "bph3_singlecap",
    "bph5_allcap",
})

#: Substrings marking the three non-phantoms and the six superseded `model_*_BPH_lesion`
#: scans. `Prostate_PerceptionFiducial` matters most: its air cavity measures 361 px in a
#: 410 px shave box (0.880), tripping the 0.80 MAX_BBOX_PROPORTION guard and defeating
#: three lumen rules at once. It is not a phantom and it produces garbage.
OUT_OF_SCOPE_MARKERS = ("perceptionfiducial", "medianlesion", "model_", "_bph_lesion")
#: Scans matching this run the lesion stage upstream, so only these can carry lesions.
#: Matched case-INSENSITIVELY: `Gelatin_Lesion_BPH_1`, `lesions_bph_2` and `FLR_Larry`
#: all differ in case, and a case-sensitive test silently misses half the set.
LESION_NAME_MARKERS = ("flr", "lesion")

BODY_HU = -500.0        # everything denser than this is phantom or table, not room air
ROI_MARGIN = 12         # voxels of slack around the body bounding box
ROI_STRIDE = 4          # the box is found on a subsampled copy; the margin absorbs it
LOCAL_Z_WIN = 15        # 4.5 mm at 0.3 mm isotropic; matches the upstream arc-snap scale
HU_SCALE = 1000.0       # divisor that puts soft tissue near zero and the shell near one


@dataclass
class Case:
    """One phantom scan with its ground truth, cropped to the scan-derived ROI."""

    stem: str
    contrast: str
    hu: np.ndarray            # (Z, Y, X) float32, Hounsfield units
    truth: np.ndarray         # (Z, Y, X) uint8 in this project's class space
    spacing: tuple[float, float, float]
    ml_per_voxel: float
    scheme: str
    roi: tuple[slice, slice, slice]
    full_shape: tuple[int, int, int]

    @property
    def volumes(self) -> dict[int, float]:
        return L.present_structures(self.truth, self.ml_per_voxel)


def load_manifest() -> dict[str, dict]:
    """The corpus manifest, keyed by case stem.

    Authoritative for three things this project cannot derive: the contrast arm, the
    hold-out flags, and `image_for_mask`. That last one matters more than it looks --
    the orientation preprocess re-frames scans, so a mask produced from a reoriented run
    sits on the reoriented grid and pairing it with the original scan reads HU from the
    wrong voxels, silently and with plausible-looking results.
    """
    import csv

    with open(MANIFEST, newline="", encoding="utf-8") as handle:
        return {row["case"]: row for row in csv.DictReader(handle)}


def case_dir(row: dict) -> Path:
    return CORPUS / ARM_DIR[row["contrast"]] / row["case"]


def runs_lesion_stage(stem: str) -> bool:
    """Whether upstream would have run the lesion stage for this case."""
    low = stem.lower()
    return any(marker in low for marker in LESION_NAME_MARKERS)


def corpus_cases(arm: str | None = None) -> list[str]:
    """In-scope case stems with both an image and a combined mask on disk."""
    out = []
    for stem, row in load_manifest().items():
        if stem in OUT_OF_SCOPE or (arm and row["contrast"] != arm):
            continue
        if any(m in stem.lower() for m in OUT_OF_SCOPE_MARKERS):
            continue
        folder = case_dir(row)
        if (folder / row["image_for_mask"]).exists() and \
                (folder / row["combined_mask"]).exists():
            out.append(stem)
    return sorted(out)


def body_roi(hu: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of the largest dense object in the scan, plus a margin.

    Derived from the image alone. Taking the largest connected component rather than the
    raw threshold bbox drops the scanner table and any stray dense debris, either of
    which would otherwise inflate the box to the full field of view.
    """
    small = hu[::ROI_STRIDE, ::ROI_STRIDE, ::ROI_STRIDE] > BODY_HU
    lab, n = ndimage.label(small)
    if n == 0:
        raise ValueError("no body found above the density threshold")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = ndimage.find_objects((lab == int(sizes.argmax())).astype(np.int32))[0]
    out = []
    for axis, sl in enumerate(biggest):
        lo = max(0, sl.start * ROI_STRIDE - ROI_MARGIN)
        hi = min(hu.shape[axis], sl.stop * ROI_STRIDE + ROI_MARGIN)
        out.append(slice(lo, hi))
    return tuple(out)


def load_case(stem: str) -> Case:
    """Read one case, resolve its label scheme, and crop both volumes to the ROI.

    Ground truth is current AutoSegCT pipeline output (scheme v3), not hand painting.
    That makes the target self-consistent and current, and removes the annotation-vintage
    confounds entirely -- at the cost that upstream's known defects are now inherited as
    truth rather than being fly errors.
    """
    row = load_manifest()[stem]
    folder = case_dir(row)
    img = sitk.ReadImage(str(folder / row["image_for_mask"]))
    gt_img = sitk.ReadImage(str(folder / row["combined_mask"]))
    if img.GetSize() != gt_img.GetSize():
        raise ValueError(
            f"{stem}: scan is {img.GetSize()} but GT is {gt_img.GetSize()}; these are "
            "different grids and pairing them reads HU from the wrong voxels")

    spacing = img.GetSpacing()
    ml_per_voxel = float(np.prod(spacing)) / 1000.0
    raw_gt = sitk.GetArrayFromImage(gt_img)
    stamp = gt_img.GetMetaData(L.SCHEME_STAMP_KEY) if gt_img.HasMetaDataKey(
        L.SCHEME_STAMP_KEY) else None
    scheme = L.infer_scheme(raw_gt, ml_per_voxel, stamp)
    truth = L.to_class_space(raw_gt, scheme)

    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    roi = body_roi(hu)
    return Case(stem=stem, contrast=row["contrast"], hu=np.ascontiguousarray(hu[roi]),
                truth=np.ascontiguousarray(truth[roi]), spacing=spacing,
                ml_per_voxel=ml_per_voxel, scheme=scheme, roi=roi,
                full_shape=truth.shape)


def local_z(hu: np.ndarray, win: int = LOCAL_Z_WIN) -> np.ndarray:
    """Per-axial-slice local z-score: (x - local mean) / local sd.

    This is the normalisation the upstream low-contrast capsule stage relies on, and the
    only one that survives a change of scanner. The filter is 2D within each slice
    because slice thickness and in-plane spacing are not equal on every scanner in this
    corpus, so a 3D window would be a different physical size per axis.
    """
    size = (1, win, win)
    mean = ndimage.uniform_filter(hu, size=size, mode="nearest")
    mean_sq = ndimage.uniform_filter(hu * hu, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (hu - mean) / (np.sqrt(var) + 1e-3)


def features(case: Case) -> np.ndarray:
    """Two-channel feature volume, (2, Z, Y, X) float32: scaled HU and local z-score."""
    return np.stack([case.hu / HU_SCALE, local_z(case.hu)]).astype(np.float32)


def sample_voxels(truth: np.ndarray, per_class: int, radius: int, depth: int,
                  rng: np.random.Generator,
                  z_range: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw up to *per_class* voxel centres from each class present.

    Balanced rather than proportional: background is ~93% of the ROI and lesions under
    0.1%, so a proportional draw would let a constant "background" prediction look good.
    Centres too close to the ROI face are dropped so every patch is fully in bounds.

    *z_range* confines the draw to the phantom slab. Without it almost every background
    sample is empty air metres from any structure, which teaches the readout to separate
    air from tissue rather than to find a boundary.
    """
    z_pad, yx_pad = depth // 2, radius
    lo, hi = z_range if z_range else (0, truth.shape[0] - 1)
    lo, hi = max(lo, z_pad), min(hi, truth.shape[0] - z_pad - 1)
    keep = np.zeros(truth.shape, dtype=bool)
    keep[lo:hi + 1,
         yx_pad:truth.shape[1] - yx_pad,
         yx_pad:truth.shape[2] - yx_pad] = True

    idx_parts, y_parts = [], []
    for cls in (L.BG, *L.STRUCTURES):
        where = np.flatnonzero((truth == cls) & keep)
        if where.size == 0:
            continue
        take = min(per_class, where.size)
        chosen = rng.choice(where, size=take, replace=False)
        idx_parts.append(chosen)
        y_parts.append(np.full(take, cls, dtype=np.uint8))
    return np.concatenate(idx_parts), np.concatenate(y_parts)


def extract_patches(feat: np.ndarray, flat_idx: np.ndarray, shape: tuple[int, int, int],
                    radius: int, depth: int) -> np.ndarray:
    """Gather (N, F) flattened patches centred on *flat_idx*.

    Offsets are computed once against the flat index rather than slicing per voxel; at a
    few hundred thousand centres the per-voxel loop dominates everything else.
    """
    z, y, x = np.unravel_index(flat_idx, shape)
    dz = np.arange(-(depth // 2), depth // 2 + 1)
    dyx = np.arange(-radius, radius + 1)
    oz, oy, ox = np.meshgrid(dz, dyx, dyx, indexing="ij")
    off = (oz.ravel() * shape[1] * shape[2] + oy.ravel() * shape[2] + ox.ravel())
    centres = (z * shape[1] * shape[2] + y * shape[2] + x)[:, None]
    gather = centres + off[None, :]
    flat = feat.reshape(feat.shape[0], -1)
    return np.concatenate([flat[c][gather] for c in range(feat.shape[0])], axis=1)


def phantom_z_range(hu: np.ndarray, frac: float = 0.10) -> tuple[int, int]:
    """Axial slice span containing the phantom, anchored on the dense capsule shell.

    The ROI box runs most of the scan depth because the holder does too, so slices
    picked evenly across it land in empty air and score against an empty ground truth.
    Soft tissue is the wrong anchor: on this rig the per-slice soft-tissue count peaks
    on the *holder*, well below the phantom. The shell at 300 HU (upstream's own
    confident-capsule seed) is counted instead, which brackets the painted capsule on
    every high-contrast case here. Uses no label information.

    High-contrast only. The low-contrast capsule has no absolute HU threshold at all and
    will need the local z-score channel to anchor this instead.
    """
    per_slice = (hu >= 300.0).sum(axis=(1, 2))
    if per_slice.max() == 0:
        return 0, hu.shape[0] - 1
    inside = np.flatnonzero(per_slice >= frac * per_slice.max())
    return int(inside[0]), int(inside[-1])


def interior_voxels(truth_shape: tuple[int, int, int], radius: int,
                    depth: int) -> np.ndarray:
    """Flat indices of every voxel whose full patch lies inside the ROI."""
    keep = np.zeros(truth_shape, dtype=bool)
    keep[depth // 2:truth_shape[0] - depth // 2,
         radius:truth_shape[1] - radius,
         radius:truth_shape[2] - radius] = True
    return np.flatnonzero(keep)
