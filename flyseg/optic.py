"""The right optic lobe (MAOL v1.1) as a retinotopic front end.

This replaces the earlier mushroom-body design. The mushroom body was the right idea for
a classifier but the wrong circuit for *images*: its input is olfactory, so the map from
a CT patch onto input neurons had to be a random projection, which was the one
unprincipled piece of the whole pipeline. The optic lobe has no such problem. It is
retinotopic, so "which neuron sees patch position (x, y)" is a lookup.

MAOL ships no neuron coordinates, so the lattice is recovered from connectivity:

1. **Columns.** The columnar cell types each appear ~892 times, one per column, and the
   within-column connections are essentially one-to-one -- measured here, an L1 cell has
   a median of exactly 1 Mi1 partner carrying 100% of its synapses onto that type. So
   columns are seeded on L1 and extended type by type with a greedy best-match that
   allows one neuron of each type per column.
2. **Retinotopy.** Columns that are neighbours on the eye share lateral connections, so
   the column-to-column adjacency graph *is* the lattice. Its mean degree comes out at
   8.0, against 6 for a perfect hexagonal sheet -- the lattice is real, not imposed.
   Laplacian eigenmaps then embed it in 2D.
3. **Equalisation.** A raw spectral embedding compresses the periphery, leaving half the
   grid empty and stacking seven columns in single cells. Radii are remapped by rank so
   density is uniform on the disc, which lifts occupancy from 53% to 77%.

The result is a genuine sampling lattice with ~892 facets, and the image is sampled by
Voronoi cell rather than onto a square grid -- which is what an ommatidium does anyway,
and it degrades gracefully where the lattice is uneven.

**The acuity limit this imposes is real and is not tunable.** ~892 columns is about a
30x30 grid. The capsule wall is ~1 mm, so for the wall to subtend even one column a
patch may span at most ~30 mm. The phantom is 155-191 mm across, so a slice needs about
6x6 patches, and the fly's output is at ~1 mm -- the interommatidial limit -- against a
0.3 mm scan. A patch covering the whole phantom would put the wall five times below
resolution and the fly would see a smudge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

MAOL_DIR = Path(r"C:\flysegct\brain_right_optic")
CACHE = MAOL_DIR / "lattice_cache.npz"

#: Sign of each predicted transmitter. Glutamate is INHIBITORY in Drosophila via the
#: GluCl- channel, the opposite of the vertebrate convention. Histamine is the
#: photoreceptor transmitter and is also inhibitory -- 590 neurons here, silently
#: dropped by any table that only knows ACh/GABA/Glu. Modulators carry no fast sign.
NT_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "HIST": -1.0,
           "DA": 0.0, "SER": 0.0, "OCT": 0.0}

#: Consensus-transmitter spellings as they appear inside the Community labels string.
CONSENSUS_NT = {"acetylcholine": "ACH", "gaba": "GABA", "glutamate": "GLUT",
                "histamine": "HIST", "dopamine": "DA", "serotonin": "SER",
                "octopamine": "OCT"}

SEED_TYPE = "L1"
#: Order matters: each type is matched against columns already populated, so the early
#: entries must be the ones with the cleanest one-to-one within-column connectivity.
COLUMN_CHAIN = ("Mi1", "L2", "L3", "L5", "C3", "T1", "L4", "Tm1", "Tm2", "Tm9",
                "Mi4", "Mi9", "Tm20", "Tm3", "Tm4", "Tm6", "C2", "T2", "T2a", "T3",
                "T4a", "T4b", "T4c", "T5a", "T5b", "T5c", "T4d", "T5d",
                "Dm2", "TmY18", "TmY5a")

#: Where the image is injected. R1-R6 are only 7 neurons in MAOL -- the motion-pathway
#: photoreceptors are essentially unreconstructed -- so injection is at the lamina
#: monopolar cells they drive, which are complete. L1 is the ON channel, L2 the OFF.
INJECT_ON = "L1"
INJECT_OFF = "L2"


@dataclass
class OpticLobe:
    """Signed connectivity plus the recovered retinotopic column lattice."""

    weights: sparse.csr_matrix     # (n_neurons, n_neurons), signed, row-normalised
    index: dict[int, int]          # root id -> matrix row
    cell_type: np.ndarray          # per-neuron primary cell type
    column_of: np.ndarray          # per-neuron column, -1 if unassigned
    lattice: np.ndarray            # (n_columns, 2) equalised unit-disc coordinates
    type_rows: dict[str, np.ndarray]   # cell type -> row indices, ordered by column
    label: str = "MAOL right optic lobe"

    @property
    def n_neurons(self) -> int:
        return self.weights.shape[0]

    @property
    def n_columns(self) -> int:
        return len(self.lattice)

    def describe(self) -> str:
        assigned = int((self.column_of >= 0).sum())
        return (f"{self.label}: {self.n_neurons} neurons, "
                f"{self.weights.nnz} signed edges, {self.n_columns} columns "
                f"({assigned} neurons placed, {len(self.type_rows)} columnar types)")


def _resolve_nt(row_nt: str, community: str) -> str:
    """Predicted transmitter, falling back to the consensus call in Community labels.

    15.5% of MAOL neurons have an empty Predicted NT type, including VS, a major
    tangential cell. Dropping them loses the sign on one neuron in six.
    """
    if isinstance(row_nt, str) and row_nt.strip():
        return row_nt.strip().upper()
    if isinstance(community, str) and "consensusNt::" in community:
        tail = community.split("consensusNt::", 1)[1].split(",", 1)[0].strip().lower()
        return CONSENSUS_NT.get(tail, "")
    return ""


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    neurons = pd.read_csv(MAOL_DIR / "neurons.csv.gz", low_memory=False)
    neurons = neurons.rename(columns={"Root ID": "id", "Primary Cell Type": "type",
                                      "Predicted NT type": "nt",
                                      "Community labels": "community"})
    neurons["nt"] = [_resolve_nt(a, b) for a, b in zip(neurons["nt"],
                                                       neurons["community"])]
    conn = pd.read_csv(MAOL_DIR / "connections_princeton.csv.gz",
                       usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    # One row per (pre, post, neuropil), not per connection: the same pair recurs across
    # neuropils and would otherwise become duplicate edges.
    conn = conn[conn["neuropil"] != "NotPrimary"]
    conn = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    return neurons, conn


def _assign_columns(neurons: pd.DataFrame, conn: pd.DataFrame) -> tuple[dict, int]:
    """Seed one column per L1 cell and extend outward, one neuron of each type per column."""
    by_type = {t: set(g["id"]) for t, g in neurons.groupby("type") if isinstance(t, str)}
    seeds = sorted(by_type.get(SEED_TYPE, ()))
    column_of = {rid: i for i, rid in enumerate(seeds)}

    pre = conn["pre_root_id"].to_numpy()
    post = conn["post_root_id"].to_numpy()
    syn = conn["syn_count"].to_numpy()

    for ctype in COLUMN_CHAIN:
        members = by_type.get(ctype)
        if not members:
            continue
        score: dict[tuple[int, int], float] = {}
        for a, b, s in zip(pre, post, syn):
            for cand, other in ((a, b), (b, a)):
                if cand in members and cand not in column_of:
                    col = column_of.get(other)
                    if col is not None:
                        key = (cand, col)
                        score[key] = score.get(key, 0.0) + s
        best: dict[int, tuple[int, float]] = {}
        for (cand, col), s in score.items():
            if cand not in best or s > best[cand][1]:
                best[cand] = (col, s)
        taken: set[int] = set()
        for cand, (col, _s) in sorted(best.items(), key=lambda kv: -kv[1][1]):
            if col not in taken:
                column_of[cand] = col
                taken.add(col)
    return column_of, len(seeds)


def _embed_lattice(column_of: dict, n_col: int, conn: pd.DataFrame) -> np.ndarray:
    """2D coordinates for each column, from the column adjacency graph.

    Columns adjacent on the eye share lateral connections, so the graph is the lattice.
    Radii are equalised by rank afterwards because a raw spectral embedding compresses
    the periphery badly.
    """
    pre = conn["pre_root_id"].to_numpy()
    post = conn["post_root_id"].to_numpy()
    syn = conn["syn_count"].to_numpy()
    rows, cols, vals = [], [], []
    for a, b, s in zip(pre, post, syn):
        ca, cb = column_of.get(a), column_of.get(b)
        if ca is not None and cb is not None and ca != cb:
            rows.append(ca)
            cols.append(cb)
            vals.append(s)
    adj = sparse.coo_matrix((vals, (rows, cols)), shape=(n_col, n_col)).tocsr()
    adj = adj + adj.T

    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    d_inv = sparse.diags(1.0 / np.sqrt(deg))
    lap = sparse.eye(n_col) - d_inv @ adj @ d_inv
    _vals, vecs = eigsh(lap, k=4, sigma=-1e-6, which="LM")
    emb = vecs[:, 1:3]

    centred = emb - np.median(emb, axis=0)
    radius = np.hypot(centred[:, 0], centred[:, 1])
    theta = np.arctan2(centred[:, 1], centred[:, 0])
    order = np.argsort(radius)
    rank = np.empty(n_col, dtype=float)
    rank[order] = np.arange(n_col)
    equalised = np.sqrt((rank + 0.5) / n_col)
    return np.stack([equalised * np.cos(theta), equalised * np.sin(theta)], axis=1)


def load_optic_lobe(use_cache: bool = True) -> OpticLobe:
    """Build (or reload) the signed network and its retinotopic lattice."""
    neurons, conn = _load_tables()
    ids = neurons["id"].to_numpy()
    index = {int(v): i for i, v in enumerate(ids)}
    cell_type = neurons["type"].fillna("").to_numpy().astype(object)

    if use_cache and CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        column_arr, lattice = blob["column_of"], blob["lattice"]
    else:
        column_map, n_col = _assign_columns(neurons, conn)
        lattice = _embed_lattice(column_map, n_col, conn)
        column_arr = np.full(len(ids), -1, dtype=np.int32)
        for rid, col in column_map.items():
            row = index.get(int(rid))
            if row is not None:
                column_arr[row] = col
        np.savez_compressed(CACHE, column_of=column_arr, lattice=lattice)

    sign = neurons["nt"].map(lambda v: NT_SIGN.get(v, 0.0)).to_numpy()
    pre_row = neurons["id"].map(index).to_numpy()
    del pre_row

    rows = conn["pre_root_id"].map(index).to_numpy()
    cols = conn["post_root_id"].map(index).to_numpy()
    good = ~(pd.isna(rows) | pd.isna(cols))
    rows = rows[good].astype(np.int64)
    cols = cols[good].astype(np.int64)
    weight = conn["syn_count"].to_numpy()[good] * sign[rows]
    keep = weight != 0.0

    w = sparse.coo_matrix((weight[keep], (rows[keep], cols[keep])),
                          shape=(len(ids), len(ids))).tocsr()
    # Normalise by each postsynaptic cell's total input so activity neither dies nor
    # blows up over the propagation steps.
    scale = np.asarray(abs(w).sum(axis=0)).ravel()
    scale[scale == 0] = 1.0
    w = w @ sparse.diags(1.0 / scale)

    type_rows: dict[str, np.ndarray] = {}
    n_col = len(lattice)
    for ctype in (SEED_TYPE, *COLUMN_CHAIN):
        rows_of_type = np.full(n_col, -1, dtype=np.int64)
        hits = np.flatnonzero((cell_type == ctype) & (column_arr >= 0))
        rows_of_type[column_arr[hits]] = hits
        if (rows_of_type >= 0).sum() >= 0.5 * n_col:
            type_rows[ctype] = rows_of_type

    return OpticLobe(weights=w.tocsr(), index=index, cell_type=cell_type,
                     column_of=column_arr, lattice=lattice, type_rows=type_rows)


def rewire(lobe: OpticLobe, rng: np.random.Generator) -> OpticLobe:
    """Degree-matched random rewiring: same degrees, same signs, scrambled topology.

    This is the control that decides what the fly's wiring is actually worth. Measuring
    the connectome as a fixed untrained transform and finding it costs accuracy does not
    by itself say anything about *flies* -- a deep fixed sparse transform with untuned
    weights generically destroys information, so the loss may be generic rather than
    specific. Comparing against a rewired network with identical degree sequence splits
    the single "network" term into "this topology" and "being a fixed sparse transform at
    all".

    Permuting the postsynaptic column preserves each neuron's in-degree exactly (every
    target appears the same number of times) while leaving the presynaptic column
    untouched, which preserves out-degree and -- because transmitter is a property of the
    sending neuron -- every edge's sign as well. Only who-connects-to-whom changes.
    """
    coo = lobe.weights.tocoo()
    shuffled = rng.permutation(coo.col)
    w = sparse.coo_matrix((coo.data, (coo.row, shuffled)), shape=coo.shape).tocsr()
    scale = np.asarray(abs(w).sum(axis=0)).ravel()
    scale[scale == 0] = 1.0
    w = (w @ sparse.diags(1.0 / scale)).tocsr()
    return OpticLobe(weights=w, index=lobe.index, cell_type=lobe.cell_type,
                     column_of=lobe.column_of, lattice=lobe.lattice,
                     type_rows=lobe.type_rows,
                     label="MAOL right optic lobe, DEGREE-MATCHED REWIRED")


def rewire_local(lobe: OpticLobe, rng: np.random.Generator,
                 radius: float | None) -> OpticLobe:
    """Scramble wiring *within* a locality and cell-type constraint.

    The global shuffle in ``rewire`` destroys retinotopic locality along with everything
    else, and for an image task locality is close to everything -- a convolution beats a
    random dense layer for exactly that reason, with no flies involved. So a win over the
    global shuffle may only say "neighbouring columns should talk to each other", which
    any retinotopic lattice delivers.

    This control holds locality and cell type fixed and destroys only the specific fly
    pattern. Neurons are **relabelled** within (cell type, spatial bin) groups: the neuron
    serving column *c* keeps its type and comes from within *radius* of *c*, but its
    connectivity is another neuron's. Relabelling is an isomorphism, so in-degree,
    out-degree and the sign structure are preserved **exactly** -- no configuration-model
    edge loss at all.

    ``radius=None`` permutes within cell type globally, which is the r -> infinity end of
    the sweep and the right comparison point for the local arms.
    """
    perm = np.arange(lobe.n_neurons, dtype=np.int64)
    for rows in lobe.type_rows.values():
        valid = np.flatnonzero(rows >= 0)
        if valid.size < 2:
            continue
        members = rows[valid]
        if radius is None:
            perm[members] = rng.permutation(members)
            continue
        bins: dict[tuple[int, int], list[int]] = {}
        for slot, col in enumerate(valid):
            key = tuple((lobe.lattice[col] / radius).astype(int))
            bins.setdefault(key, []).append(slot)
        for slots in bins.values():
            if len(slots) < 2:
                continue
            group = members[np.asarray(slots)]
            perm[group] = rng.permutation(group)

    w = lobe.weights[perm][:, perm].tocsr()
    label = ("MAOL right optic lobe, TYPE-PRESERVING GLOBAL SHUFFLE" if radius is None
             else f"MAOL right optic lobe, LOCAL SHUFFLE r={radius:g}")
    return OpticLobe(weights=w, index=lobe.index, cell_type=lobe.cell_type,
                     column_of=lobe.column_of, lattice=lobe.lattice,
                     type_rows=lobe.type_rows, label=label)


def sampling_map(lattice: np.ndarray, patch_px: int) -> np.ndarray:
    """For each pixel of a patch_px x patch_px patch, the column that sees it.

    Voronoi assignment on the lattice rather than a square grid: the facets are not on a
    regular grid and forcing one would leave a quarter of the cells empty.
    """
    axis = (np.arange(patch_px) + 0.5) / patch_px * 2.0 - 1.0
    gy, gx = np.meshgrid(axis, axis, indexing="ij")
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
    _dist, nearest = cKDTree(lattice).query(pix, k=1)
    return nearest.reshape(patch_px, patch_px).astype(np.int32)
