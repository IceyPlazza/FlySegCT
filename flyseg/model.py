"""The classifier: real optic-lobe wiring, one trained readout.

This is a rebuild. The earlier version projected a CT patch randomly onto mushroom-body
input neurons, because the mushroom body's own input is olfactory and there was no
principled map from an image. The optic lobe needs no such invention -- it is
retinotopic, so the patch lands on the lattice directly.

    patch -> Voronoi pooling onto ~892 facets
          -> ON/OFF split, injected at L1 and L2
          -> T steps through the signed optic-lobe network
          -> per-column activity of the columnar cell types
          -> shared ridge readout -> a class per column

Only the readout is fitted, and it is **shared across columns** -- the same readout at
every facet, which is what the lobe's own repeated circuit does and what makes the model
translation-invariant rather than memorising positions.

Two consequences worth stating plainly.

**The output is at column resolution, ~1 mm, against a 0.3 mm scan.** That is the
interommatidial limit and it is a hard ceiling, not a tuning choice. Predictions are
upsampled back to voxels for scoring, so every score carries that quantisation.

**This segments a patch, not a voxel.** One forward pass labels all ~892 facets at once
instead of one voxel, which is what makes whole-slice evaluation affordable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from . import labels as L

#: Propagation steps. The lamina-to-lobula path is about four synapses deep, so four
#: steps is roughly one pass through the lobe rather than an arbitrary depth.
STEPS = 4

#: Patch extent in millimetres. Capped by acuity: ~892 columns is about a 30x30 grid and
#: the capsule wall is ~1 mm, so beyond ~30 mm the wall stops subtending a single column
#: and the fly sees a smudge. Not tunable upward without losing the wall.
PATCH_MM = 30.0

RIDGE_LAMBDA = 1.0
PATCH_BATCH = 32


@dataclass
class OpticSegmenter:
    """Fixed optic-lobe network plus a fitted per-column readout.

    Setting ``bypass=True`` skips the network entirely and hands the readout the raw
    pooled facet values instead. That is the control that decides what the connectome is
    actually contributing: identical lattice, identical pooling, identical output
    resolution, identical readout and training data -- the only difference is whether the
    features came from 52,445 neurons or straight off the image.
    """

    lobe: object
    steps: int = STEPS
    bypass: bool = False
    #: NOTE: bypass must receive the SAME encoded input the network does -- the ON/OFF
    #: contrast pair plus the local-z channel -- not the raw pooled values. Feeding it
    #: absolute HU would make bypass-vs-fly differ in both the network AND the encoding,
    #: so any gap could no longer be attributed to the network. Same input both sides is
    #: the whole point of the control.
    #: Exponent on the inverse-frequency class weight. 1.0 fully balances the classes and
    #: 0.0 leaves them at their natural prior. Full balancing is what makes rare classes
    #: detectable and is also what makes every learned arm over-predict by one to two
    #: orders of magnitude, so it is the dominant calibration knob here.
    weight_power: float = 1.0
    readout: np.ndarray | None = field(default=None, init=False)
    classes: np.ndarray | None = field(default=None, init=False)
    feature_types: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.feature_types = tuple(sorted(self.lobe.type_rows))

    @property
    def n_features(self) -> int:
        return 3 if self.bypass else len(self.feature_types) * 2

    def encoded_input(self, facet: np.ndarray) -> np.ndarray:
        """The network's own drive, per facet: (ON, OFF, local z).

        Exactly what ``run_network`` injects, before any propagation. Handing this to the
        readout isolates the network as the single difference.
        """
        contrast = facet[:, :, 0] - facet[:, :, 0].mean(axis=1, keepdims=True)
        return np.stack([np.maximum(contrast, 0.0), np.maximum(-contrast, 0.0),
                         facet[:, :, 1]], axis=-1).astype(np.float32)

    def pooling_operator(self, sample_map: np.ndarray):
        """Sparse (pixels x facets) averaging matrix, built once per patch size.

        A scatter-add over pixels is the obvious way to do this and is far too slow at
        ten thousand pixels a patch; as a sparse matmul it is one BLAS call.
        """
        key = sample_map.shape
        if getattr(self, "_pool_key", None) == key:
            return self._pool_op
        idx = sample_map.ravel().astype(np.int64)
        n_col = self.lobe.n_columns
        counts = np.bincount(idx, minlength=n_col).astype(np.float32)
        counts[counts == 0] = 1.0
        op = sparse.coo_matrix(
            (1.0 / counts[idx], (np.arange(idx.size), idx)),
            shape=(idx.size, n_col), dtype=np.float32).tocsr()
        self._pool_key, self._pool_op = key, op
        return op

    def pool(self, patches: np.ndarray, sample_map: np.ndarray) -> np.ndarray:
        """Average each channel over every facet's Voronoi cell -> (B, n_col, C)."""
        b, _py, _px, chan = patches.shape
        op = self.pooling_operator(sample_map)
        flat = patches.reshape(b, -1, chan)
        out = np.empty((b, self.lobe.n_columns, chan), dtype=np.float32)
        for c in range(chan):
            out[:, :, c] = flat[:, :, c] @ op
        return out

    def run_network(self, facet: np.ndarray) -> np.ndarray:
        """Propagate facet drive through the lobe and return per-column activity.

        The ON/OFF split is contrast against the patch mean, which is what the lamina
        actually encodes: L1 carries increments, L2 decrements. Feeding raw intensity
        instead would make the response depend on the patch's absolute level, which the
        real circuit removes on purpose.
        """
        b, n_col, _chan = facet.shape
        contrast = facet[:, :, 0] - facet[:, :, 0].mean(axis=1, keepdims=True)
        drive = np.zeros((b, self.lobe.n_neurons), dtype=np.float32)

        on_rows = self.lobe.type_rows.get("L1")
        off_rows = self.lobe.type_rows.get("L2")
        for rows, signal in ((on_rows, np.maximum(contrast, 0.0)),
                             (off_rows, np.maximum(-contrast, 0.0))):
            if rows is None:
                continue
            valid = rows >= 0
            drive[:, rows[valid]] += signal[:, valid]
        # The local z-score channel rides in alongside, at the same facets: it is the
        # only feature that survives a change of scanner.
        zc = facet[:, :, 1]
        if on_rows is not None:
            valid = on_rows >= 0
            drive[:, on_rows[valid]] += 0.5 * zc[:, valid]

        # Transposed sparse-times-dense rather than dense-times-sparse: the latter goes
        # through scipy's __rmatmul__ and comes back as np.matrix, which silently breaks
        # the shapes downstream.
        w_t = self.lobe.weights.T.tocsr()
        state = np.zeros_like(drive)
        for _ in range(self.steps):
            state = np.tanh((w_t @ state.T).T + drive)

        feats = np.empty((b, n_col, self.n_features), dtype=np.float32)
        for i, ctype in enumerate(self.feature_types):
            rows = self.lobe.type_rows[ctype]
            valid = rows >= 0
            col = np.zeros((b, n_col), dtype=np.float32)
            col[:, valid] = state[:, rows[valid]]
            feats[:, :, 2 * i] = col
            feats[:, :, 2 * i + 1] = np.where(valid, 1.0, 0.0)
        return feats

    def features(self, patches: np.ndarray, sample_map: np.ndarray) -> np.ndarray:
        facet = self.pool(patches, sample_map)
        return self.encoded_input(facet) if self.bypass else self.run_network(facet)

    def fit(self, feature_batches, classes: np.ndarray) -> None:
        """Fit the shared readout from an iterable of (features, column labels).

        Samples are weighted by inverse class frequency. Background dominates the facets
        by an order of magnitude and an unweighted fit simply predicts it everywhere.
        """
        self.classes = np.asarray(classes, dtype=np.uint8)
        dim, n_cls = self.n_features + 1, len(self.classes)
        gram = np.zeros((dim, dim), dtype=np.float64)
        rhs = np.zeros((dim, n_cls), dtype=np.float64)
        seen = np.zeros(n_cls, dtype=np.float64)

        cached = []
        for feats, y in feature_batches:
            flat = feats.reshape(-1, self.n_features)
            cached.append((flat, y.ravel()))
            seen += np.bincount(np.searchsorted(self.classes, y.ravel()),
                                minlength=n_cls)

        freq = np.where(seen > 0, seen, 1.0)
        weight_of = (freq.sum() / (n_cls * freq)) ** self.weight_power

        for flat, y in cached:
            design = np.hstack([flat, np.ones((len(flat), 1), dtype=np.float32)])
            slot = np.searchsorted(self.classes, y)
            w = weight_of[slot].astype(np.float64)
            target = (slot[:, None] == np.arange(n_cls)[None, :]).astype(np.float64)
            gram += (design * w[:, None]).T @ design
            rhs += (design * w[:, None]).T @ target

        reg = RIDGE_LAMBDA * np.eye(dim)
        reg[-1, -1] = 0.0
        self.readout = np.linalg.solve(gram + reg, rhs)

    def predict_columns(self, feats: np.ndarray) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("readout has not been fitted")
        flat = feats.reshape(-1, self.n_features)
        design = np.hstack([flat, np.ones((len(flat), 1), dtype=np.float32)])
        best = np.argmax(design @ self.readout, axis=1)
        return self.classes[best].reshape(feats.shape[:2])


#: Upstream's high-contrast lesion gates, applied last, after everything else. The count
#: cap -- not threshold or margin tuning -- is what removes fiducials in the real
#: pipeline, so it is convention rather than tuning-to-fit.
LESION_MIN_MM3 = 216.0
LESION_MAX_MM3 = 5400.0
LESION_MAX_COUNT = 3


def facet_threshold(facet_hu: np.ndarray) -> np.ndarray:
    """The same HU rule, but reading the fly's own input: pooled HU at facet resolution.

    The full-resolution threshold reads raw HU at 0.3 mm while the fly reads pooled
    values at ~1 mm, so comparing them charges the connectome for a resolution handicap
    it did not choose. This baseline removes exactly that difference and nothing else.
    """
    out = np.zeros(facet_hu.shape, dtype=np.uint8)
    out[facet_hu < -450.0] = L.LUMEN
    out[(facet_hu >= -200.0) & (facet_hu < 240.0)] = L.LOBE
    out[facet_hu >= 300.0] = L.CAPSULE
    return out


@dataclass
class EncodedThreshold:
    """A threshold on the fly's ACTUAL input: facet-pooled, patch-mean contrast.

    This is the arm that separates the two remaining hypotheses. ``facet_threshold``
    pools to facet resolution but keeps absolute HU, so it tests resolution only -- and
    resolution turned out to help. The fly never sees absolute HU: it gets ON/OFF
    contrast against the patch mean, which discards the absolute level by construction.

    So this thresholds exactly what the fly is given. Three scalars are fitted, one
    boundary per structure, which is the least capacity that can express a threshold at
    all -- far below the readout's 65 parameters. If this scores near the absolute-HU
    arm, the encoding is cheap and the architecture carries the deficit; if it scores
    near the fly, the encoding carries it.
    """

    t_capsule: float = 0.0
    t_lumen: float = 0.0
    t_lobe: float = 0.0

    @staticmethod
    def contrast(facet: np.ndarray) -> np.ndarray:
        """The same ON/OFF drive the network receives, before propagation."""
        return facet[:, :, 0] - facet[:, :, 0].mean(axis=1, keepdims=True)

    def fit(self, contrast: np.ndarray, target: np.ndarray) -> "EncodedThreshold":
        flat, y = contrast.ravel(), target.ravel()
        grid = np.quantile(flat, np.linspace(0.001, 0.999, 200))

        def best(mask: np.ndarray, above: bool) -> float:
            if not mask.any():
                return float("inf") if above else float("-inf")
            scores = []
            for t in grid:
                pick = flat >= t if above else flat <= t
                inter = float(np.count_nonzero(pick & mask))
                total = float(np.count_nonzero(pick) + np.count_nonzero(mask))
                scores.append(0.0 if total == 0 else 2.0 * inter / total)
            return float(grid[int(np.argmax(scores))])

        self.t_capsule = best(y == L.CAPSULE, above=True)
        self.t_lumen = best(y == L.LUMEN, above=False)
        self.t_lobe = best(y == L.LOBE, above=True)
        return self

    def predict(self, contrast: np.ndarray) -> np.ndarray:
        out = np.zeros(contrast.shape, dtype=np.uint8)
        out[contrast >= self.t_lobe] = L.LOBE
        out[contrast <= self.t_lumen] = L.LUMEN
        out[contrast >= self.t_capsule] = L.CAPSULE
        return out

    def describe(self) -> str:
        return (f"capsule>={self.t_capsule:+.3f} lumen<={self.t_lumen:+.3f} "
                f"lobe>={self.t_lobe:+.3f}")


def gate_lesions(pred: np.ndarray, mm3_per_voxel: float,
                 open_voxels: int = 2) -> tuple[np.ndarray, str]:
    """Apply upstream's size band and count cap to the lesion class.

    Returns the gated map and an advisory. The cap is loud on purpose: dropping a real
    lesion because a fourth component outranked it is a thing a human needs to know about,
    and silently keeping three is how that goes unnoticed.

    Only meaningful when the prediction covers a contiguous slab. On isolated slices the
    connected components are 2D islands, not lesions, and a 3D size band does not apply.
    """
    from scipy import ndimage

    out = pred.copy()
    raw = pred == L.LESION
    _lab0, n_raw = ndimage.label(raw)
    # Open BEFORE gating. The prediction is piecewise constant on ~1 mm facets but is
    # counted on the 0.3 mm scan grid, so one noisy facet fragments into many specks and
    # inflates the component count. Gating first is what let dust outrank a true positive.
    if open_voxels > 0:
        raw = ndimage.binary_opening(raw, np.ones((1, open_voxels, open_voxels)))
    lab, n = ndimage.label(raw)
    if n == 0:
        return out, f"{n_raw} raw components, none survived opening"

    sizes = np.bincount(lab.ravel())[1:] * mm3_per_voxel
    in_band = [i + 1 for i, mm3 in enumerate(sizes)
               if LESION_MIN_MM3 <= mm3 <= LESION_MAX_MM3]
    dropped_band = n - len(in_band)

    in_band.sort(key=lambda i: -sizes[i - 1])
    kept = in_band[:LESION_MAX_COUNT]
    capped = len(in_band) - len(kept)

    out[(pred == L.LESION) & ~np.isin(lab, kept)] = L.BG
    advisory = (f"{n_raw} raw -> {n} after opening -> {dropped_band} outside "
                f"{LESION_MIN_MM3:.0f}-{LESION_MAX_MM3:.0f} mm3, "
                f"{len(in_band)} in band, kept {len(kept)}")
    if capped:
        advisory += f"  ** ADVISORY: count cap dropped {capped} in-band component(s) **"
    return out, advisory


def threshold_baseline(hu: np.ndarray) -> np.ndarray:
    """A fixed-HU-threshold baseline. **This is NOT the AutoSegCT pipeline.**

    The constants are upstream's (300 HU confident shell, 240 HU thin-edge floor,
    -450 HU air), but the real capsule stage does not threshold on HU: it band-selects
    whole connected components that touch a band and then edge-trims. Plain hysteresis
    at a fixed floor floods straight through the lobe, which is exactly why the project
    abandoned it. Reporting this number as "AutoSegCT" would claim a comparison that has
    not been run.
    """
    out = np.zeros(hu.shape, dtype=np.uint8)
    out[hu < -450.0] = L.LUMEN
    out[(hu >= -200.0) & (hu < 240.0)] = L.LOBE
    out[hu >= 300.0] = L.CAPSULE
    return out
