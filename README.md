# FlySegCT

Segmenting CT scans of BPH prostate phantoms using the wiring diagram of a fruit fly.

This is a joke, run honestly. The connectivity is real (FlyWire *Drosophila* connectome),
the label handling follows the conventions of the parent AutoSegCT project, and the
scores are computed the way that project computes them — so when a number comes out bad,
it is bad for a reason you can name rather than because the rig was wrong.

## The idea

Use the fly's **visual system** to look at a CT slice. The right optic lobe is
retinotopic: it is a lattice of ~892 columns, each looking at one direction, wired into
edge and motion circuitry (L1–L5, Mi, Tm, and the T4/T5 motion detectors). So "which
neuron sees this part of the image" is a lookup rather than an invention.

```
CT patch ──▶ Voronoi pooling onto ~892 facets
         ──▶ ON/OFF contrast, injected at L1 and L2
         ──▶ 4 steps through the signed optic-lobe network
         ──▶ per-column activity of 32 columnar cell types
         ──▶ trained readout, shared across columns ──▶ a class per facet
```

Only the readout is fitted, and it is the *same* readout at every facet — which is what
the lobe's own repeated circuit does, and what stops the model memorising positions.

The connectome ships no neuron coordinates, so the lattice is recovered from connectivity
alone: columns from the one-to-one within-column wiring (an L1 cell has a median of
exactly one Mi1 partner carrying 100% of its synapses), and retinotopy from the
column-to-column adjacency graph, whose mean degree comes out at **8.0** against 6 for a
perfect hexagonal sheet.

### The acuity ceiling

~892 facets is about a 30×30 grid, and the capsule wall is ~1 mm — so a patch can span at
most ~30 mm before the wall stops subtending a single column. The phantom is 155–191 mm
across, so a slice takes ~6×6 patches and **the fly sees at ~1 mm against a 0.3 mm scan**.
That is the interommatidial limit. It is not tunable, and every score below carries it.

## What it is asked to segment

Four structures in a CT scan of a physical benign-prostatic-hyperplasia phantom:
**capsule** (a dense shell), **lumen** (an air-filled triradiate tube), **lobe** (tissue
filling the space between them), and focal **lesions** (dense inclusions in the lobe).

Two of these are not recoverable from local appearance by *any* classifier, and the
project says so up front rather than quietly scoring badly:

- The **lumen is air, and so is the room** — their median Hounsfield values differ by 14
  HU. The lumen is defined by being *inside the body*, not by how it looks.
- The **lobe has no intensity definition at all**. Upstream it is literally
  `fill(capsule interior) − capsule − lumen`.

So **capsule** and **lesion** are the real targets, and **lobe** and **lumen** are carried
as known-impossible controls. A terrible lobe score is then a specific finding — that the
fly fails exactly where a threshold fails — instead of a shrug.

## Running it

```bash
python run_flyseg.py --test bph_model1_08_14_25
```

Trains the readout on the five other high-contrast cases and scores on the held-out one.
The comparison baseline is the parent pipeline's own HU constants (300 HU for the
confident shell, 240 HU thin-edge floor, −450 HU for air), not a straw man.

### Connectome data

Codex exports for **MAOL v1.1** (Male Adult Optic Lobe, Right) go in `brain_right_optic/`:

```
brain_right_optic/neurons.csv.gz               52,445 neurons
brain_right_optic/connections_princeton.csv.gz 6,736,968 rows
```

Downloads need a (free) login at [codex.flywire.ai](https://codex.flywire.ai/). The first
run builds and caches the column lattice (~2 minutes); later runs load it in seconds.

## Results

Test case held out from training, contiguous 40-slice slab, 30 mm patches:

| structure | fly | fixed HU threshold | all-lobe | random (prior) |
|---|---|---|---|---|
| capsule | 0.212 | **0.526** | 0.000 | 0.020 |
| lumen | 0.146 | **0.311** | 0.000 | 0.034 |
| lobe | 0.212 | 0.273 | **0.241** | 0.137 |
| lesion, per component | 0.157 / 0.676 | all missed | — | — |

**The fly loses to a fixed HU threshold on every structure.** That is the honest result,
and it is roughly what the data predicts: two of the four targets have no local
appearance to learn, and the fly pays a 3.3× resolution penalty on top.

The degenerate columns are the reason those numbers can be read at all. "Predict lobe
everywhere" scores 0.241 — above the fly and just below the threshold — so on the lobe
*neither method meaningfully beats a constant*, which is what you expect of a structure
defined by subtraction rather than appearance.

Two earlier, better-looking results were retracted after evaluating on a contiguous slab
instead of scattered slices; the details are in the project notes. The short version: on
scattered slices, connected components are 2D islands rather than solids, which flatters
per-component lesion scores and starves the lobe of ground truth to score against.

## Layout

```
flyseg/labels.py      label schemes and the mapping into one class space
flyseg/data.py        loading, ROI cropping, normalisation, patch sampling
flyseg/optic.py       MAOL → signed network + recovered retinotopic lattice
flyseg/model.py       facet pooling, network propagation, shared ridge readout
flyseg/score.py       Dice, per-component lesion Dice, symmetric difference
run_flyseg.py         end-to-end train/test
```

## Requirements

`numpy`, `scipy`, `SimpleITK`.

## A note on the data

This project reads the phantom corpus and ground truth read-only and writes nothing back
to either the parent AutoSegCT repo or the nnU-Net sibling. The scans are not distributed
with it.
