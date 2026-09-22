"""Label handling for BPH prostate phantom masks.

Two different label schemes arrive in this project and they disagree about what the
integers 4 and 5 mean. Getting that wrong silently swaps "lesion" for "lobe" and
produces a plausible-looking Dice against the wrong structure, so every mask read here
is resolved to a scheme first and mapped into one common class space second.

The schemes (upstream AutoSegCT names them bph-v1 / bph-v2 / bph-v3):

    bph-v1   lobe whole at 3, lesions contiguous from 4
    bph-v2   lobe partitioned 3/4/5, lesions contiguous from 6
    bph-v3   lobe partitioned 3/4/5, lesions keyed to the lobe SECTOR they sit in,
             so the lesion range is sparse -- {7} alone and {6, 8} are both legal

Every hand-painted combined ground truth in this corpus is effectively v1, while the
pipeline emits v3 today. The header stamp that would settle it lives in ITK_FileNotes,
but annotation tools blank it on save -- the starter case's GT has an empty stamp -- so
inference from label volumes is the normal path, not the fallback.

v2 and v3 are not separable by values or geometry. That does not matter here because
this project collapses all lesions into one class anyway.
"""

from __future__ import annotations

import numpy as np

SCHEME_V1 = "bph-v1"
SCHEME_V2 = "bph-v2"
SCHEME_V3 = "bph-v3"
SCHEME_PARTITIONED = "bph-v2/v3"
KNOWN_SCHEMES = (SCHEME_V1, SCHEME_V2, SCHEME_V3)

SCHEME_STAMP_KEY = "ITK_FileNotes"
SCHEME_STAMP_PREFIX = "autosegct-"

# The class space this project works in. Mirrors the sibling nnU-Net target scheme so a
# number computed here is comparable to one computed there.
BG = 0
CAPSULE = 1
LUMEN = 2
LOBE = 3
LESION = 4
CLASS_NAMES = {BG: "background", CAPSULE: "capsule", LUMEN: "lumen",
               LOBE: "lobe", LESION: "lesion"}
STRUCTURES = (CAPSULE, LUMEN, LOBE, LESION)

#: A painted lobe part is a sector of a ~100 mL lobe; a v3 lesion slot holds 1.4-3.6 mL.
#: Volume, not label value, is what separates the two readings.
PAINTED_LOBE_MIN_ML = 10.0

#: Below this a "structure" is annotation residue rather than a painted object. Scoring
#: against it yields a Dice computed on noise.
NEGLIGIBLE_ML = 0.1


class SchemeError(ValueError):
    """Raised when a mask cannot be resolved to a known label scheme."""


def scheme_from_stamp(stamp: str | None) -> str | None:
    """The scheme a header stamp names, or None for a blank or unrecognised one.

    Parsed into a version, never compared for equality against the current stamp:
    an equality test reclassifies every previously-stamped file as current the moment
    upstream's SCHEME_CURRENT moves.
    """
    if not stamp:
        return None
    for scheme in KNOWN_SCHEMES:
        if stamp == f"{SCHEME_STAMP_PREFIX}{scheme}":
            return scheme
    return None


def infer_scheme(arr: np.ndarray, ml_per_voxel: float, stamp: str | None = None) -> str:
    """Resolve *arr* to a label scheme, preferring volume evidence over the stamp.

    Volume outranks the stamp deliberately. An annotator repainting a pipeline mask
    keeps its header, so an inherited v3 stamp on hand-painted v1 content is a real
    scenario in this corpus and the stamp is the thing that lies.
    """
    present = set(np.unique(arr).tolist()) - {0}
    if not present:
        raise SchemeError("mask is empty")

    def ml(label: int) -> float:
        return float((arr == label).sum()) * ml_per_voxel

    # Painted lobe parts at 7/8/9 predate every code scheme. v3 now uses those same
    # integers for lesion slots, so only size tells the two apart.
    if any(ml(v) >= PAINTED_LOBE_MIN_ML for v in (7, 8, 9)):
        raise SchemeError(
            "mask carries labels 7/8/9 above the painted-lobe volume threshold; this is "
            "hand-painted lobe partition, not a v3 lesion slot, and needs an explicit map")

    if not present & {4, 5}:
        # No 4 or 5 at all: the lobe is whole and any lesions sit at 6+, so the v1 and
        # partitioned readings agree on every voxel.
        return SCHEME_V1 if not present & {6, 7, 8, 9} else SCHEME_PARTITIONED

    lobe_sized = max(ml(4), ml(5)) >= PAINTED_LOBE_MIN_ML
    if lobe_sized:
        return SCHEME_PARTITIONED
    stamped = scheme_from_stamp(stamp)
    if stamped in (SCHEME_V2, SCHEME_V3):
        raise SchemeError(
            f"stamp says {stamped} but labels 4/5 hold only "
            f"{max(ml(4), ml(5)):.2f} mL, far below a lobe part; refusing to guess")
    return SCHEME_V1


def to_class_space(arr: np.ndarray, scheme: str) -> np.ndarray:
    """Map a mask in *scheme* onto this project's five-class space.

    Lesions collapse to one class because the upstream schemes emit one integer per
    lesion, which is open-ended; individual lesions are recovered by connected
    components at scoring time instead.
    """
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr == 1] = CAPSULE
    out[arr == 2] = LUMEN
    if scheme == SCHEME_V1:
        out[arr == 3] = LOBE
        out[arr >= 4] = LESION
    elif scheme in (SCHEME_V2, SCHEME_V3, SCHEME_PARTITIONED):
        out[(arr >= 3) & (arr <= 5)] = LOBE
        out[arr >= 6] = LESION
    else:
        raise SchemeError(f"no mapping rules for scheme {scheme!r}")
    return out


def present_structures(arr: np.ndarray, ml_per_voxel: float) -> dict[int, float]:
    """Volumes in mL of the structures genuinely present in a class-space mask.

    Anything under NEGLIGIBLE_ML is dropped rather than reported as present.
    """
    out = {}
    for label in STRUCTURES:
        vol = float((arr == label).sum()) * ml_per_voxel
        if vol >= NEGLIGIBLE_ML:
            out[label] = vol
    return out
