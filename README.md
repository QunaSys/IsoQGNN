# IsoQGNN: Topology-Aligned Quantum and Classical GNNs for Molecular Property Prediction

Companion code for:

> **Implementations of Quantum and Classical Topology-Aligned Architectures for Molecular Property Prediction**
> James T. Pegg, Hubert Okadome Valencia, Ronin Wu.
> IEEE International Conference on Quantum Computing and Engineering (QCE26), Toronto, 2026.

Presented at IEEE Quantum Week 2026 (QCE26) in Toronto on 15 September 2026,
in the *Quantum ML for Chemistry and Molecules* session.

This repository implements a **topology-aligned inductive bias** for molecular
property prediction, in which the model architecture mirrors the molecular bond
graph: atoms map to a fixed register of computational units, and bonds determine
which pairs interact through shared, learnable parameters. The principle is
instantiated in two parameter-matched (64-parameter) architectures:

- **Iso-QGNN** — a variational quantum circuit where entangling operations act
  between qubits corresponding to bonded atoms ([qgnn.py](qgnn.py)).
- **Iso-CGNN** — a classical message-passing network that mirrors Iso-QGNN
  primitive-by-primitive at identical parameter count ([cgnn.py](cgnn.py)).

Both are benchmarked on QM9 binary classification of the HOMO-LUMO gap and the
electric dipole moment, using byte-identical data splits and matched RNG state so
that the comparison isolates the *substrate* (quantum vs. classical) from the
*architecture*.

## Installation

Requires Python 3.11; the pinned dependencies in `requirements.txt` were
tested with this version only.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

QM9 is downloaded automatically from the DeepChem S3 mirror on first run; no
manual data download is needed.

## Reproducing the paper

Each figure and table is produced by a single script. All quantum training is
noiseless PennyLane state-vector simulation on a CPU: roughly 33 minutes per
trial per task for Iso-QGNN against 2 minutes for Iso-CGNN, so
`run_experiments.py` takes about 12 hours for ten trials across both tasks,
excluding the QM9 download and plotting.

| Paper artifact | Script | Output |
| --- | --- | --- |
| Table III (test performance at matched capacity), Fig. 2 (confusion), Fig. 4 (gradients), Fig. 5 (learned bond parameters) | [run_experiments.py](run_experiments.py) | `experiment_summary_metrics.csv`, `experiment_full_curves.csv`, `experiment_confusion_matrices.png`, `experiment_gradient_stability.png`, `final_bonds_errorbars.png`, plus two supplementary plots (`experiment_metrics_mean.png`, `experiment_loss_curves_mean.png`) |
| Table III (non-graph LogReg control row) | [run_trivial_baseline.py](run_trivial_baseline.py) | `trivial_baseline_metrics.csv` |
| Fig. 3 (sample complexity) | [run_data_efficiency.py](run_data_efficiency.py) | `data_efficiency_combined.csv`, `.png` |

```bash
python run_experiments.py       # main matched comparison (10 seeds)
python run_trivial_baseline.py  # non-graph logistic-regression control
python run_data_efficiency.py   # training-set-size sweep
```

The matched comparison and the non-graph control use ten trials with seeds
`{42, 43, …, 51}`; within each trial the 80/10/10 split is drawn once and shared
by both architectures. The data-efficiency sweep uses five trials per training
size; within each trial the training subset is drawn once from the pool and
shared by both architectures. In every case all RNGs are reset to a common seed
before each model is built.

**Reproducibility note.** The variational quantum models are sensitive to
initialisation — most visibly on the gap task, where individual seeds can
differ by up to ~0.15 AUC and under-trained runs can leave a seed unconverged.
Results are therefore reported over ten seeds at convergence (200 epochs in
`run_experiments.py`), and `requirements.txt` pins exact dependency versions.
Every metric from the matched comparison is the final-epoch value, with no
model selection of any kind.
The data-efficiency sweep selects each epoch by validation AUC (never the test
set) and uses a shorter 40-epoch budget per point for tractability across the
19 training sizes.

## Task definition

Both targets are continuous QM9 scalars converted to binary labels by a
**median split**. A fixed 1,500-molecule sample (seed 42) is drawn once. Within
each trial the 80/10/10 split is drawn, the threshold is set to the median of
the 1,200-molecule *training* split only, and a molecule is labelled 1 if its
target exceeds that threshold and 0 otherwise. The same threshold is applied
unchanged to the validation and test molecules, so the class boundary never
sees held-out targets. The training set is balanced by construction (600/600,
up to ties at the median) and the held-out splits are close to balanced. The
data-efficiency sweep uses a larger 3,000-molecule pool drawn with the same
seed and thresholds once per task on the median of its training pool, excluding
the fixed 500-molecule test set. Molecules are restricted to ≤ 9 heavy atoms
drawn from {C, N, O, F}; hydrogens are implicit.

## Repository layout

| File | Role |
| --- | --- |
| [config.py](config.py) | Global hyperparameters and structural constants (`N_MAX`, atom types, etc.) |
| [preprocessing.py](preprocessing.py) | SMILES → (atoms, bonds) graph conversion via RDKit |
| [dataset.py](dataset.py) | `MoleculeDataset` wrapper |
| [qgnn.py](qgnn.py) | Iso-QGNN variational quantum circuit (PennyLane) |
| [cgnn.py](cgnn.py) | Iso-CGNN parameter-matched classical analogue |
| [q_trainer.py](q_trainer.py) | Training loop with per-group gradient-norm tracking |
| [run_experiments.py](run_experiments.py) | Full matched comparison across ten seeds |
| [run_data_efficiency.py](run_data_efficiency.py) | Training-set-size sweep |
| [run_trivial_baseline.py](run_trivial_baseline.py) | Non-graph logistic-regression control |

## Tests

A small suite of fast invariant checks guards the properties the paper relies
on — the matched 64-parameter capacity, the single-layer gate count, the graph
construction, the train-split median labelling, and the protocol invariants
(200-epoch training over ten trials, validation-based epoch selection in the
sweep, train-only thresholds, exact dependency pins):

```bash
pip install -r requirements-dev.txt
pytest
```

They run in a few seconds and require no dataset download.

## Citation

Preprint: [arXiv:2607.13737](https://arxiv.org/abs/2607.13737).

```bibtex
@inproceedings{pegg2026isoqgnn,
  title     = {Implementations of Quantum and Classical Topology-Aligned
               Architectures for Molecular Property Prediction},
  author    = {Pegg, James T. and Okadome Valencia, Hubert and Wu, Ronin},
  booktitle = {IEEE International Conference on Quantum Computing and
               Engineering (QCE)},
  year      = {2026},
  note      = {arXiv:2607.13737},
}
```

## License

Released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
