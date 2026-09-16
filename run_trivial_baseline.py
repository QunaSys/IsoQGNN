# Copyright 2026 James T. Pegg, Hubert Okadome Valencia, and Ronin Wu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trivial Non-Graph Baseline: Logistic Regression on Composition Features
=======================================================================

This control establishes what heavy-atom composition alone can achieve on the
two tasks, so the margin above it bounds how much of the reported performance
could be attributed to easy task geometry rather than to graph structure.

Protocol is matched exactly to run_experiments.py:
  - Same 1500-molecule QM9 sample (seed 42), median-split binary labels.
  - Same ten trials with byte-identical train/val/test splits, reproduced
    via torch.random_split with generator seed 42 + trial.
  - Same metrics: test AUC-ROC and accuracy, mean +/- std across trials.

Feature set: heavy-atom counts per type (C, N, O, F) -> 4 features,
5 trainable parameters (weights + bias), a strict non-graph control.
"""

import io

import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import random_split

from config import ATOM_TYPES, N_MAX

N_TRIALS = 10

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
print("Downloading QM9...")
RAW_DF = pd.read_csv(io.StringIO(requests.get(URL).content.decode("utf-8")))


def atom_counts(smiles):
    """Composition feature vector: heavy-atom counts per type."""
    mol = Chem.MolFromSmiles(smiles)
    counts = np.zeros(len(ATOM_TYPES))
    for atom in mol.GetAtoms():
        counts[ATOM_TYPES.index(atom.GetSymbol())] += 1
    return counts


def get_arrays(target_col, sample_size=1500, seed=42):
    """Same sampling and filtering as run_experiments.py; returns composition
    features and the CONTINUOUS target. Binary labels are assigned per trial
    with the train-split median (matching the main run), not here."""
    df = RAW_DF.sample(n=sample_size, random_state=seed)
    df = df.reset_index(drop=True)

    feats, vals = [], []
    for smi, val in zip(df["smiles"], df[target_col].values):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        if len(atoms) > N_MAX or not all(a in ATOM_TYPES for a in atoms):
            continue
        feats.append(atom_counts(smi))
        vals.append(float(val))

    return np.array(feats), np.array(vals)


results = []
for prop in ["gap", "mu"]:
    X, vals = get_arrays(prop)
    n = len(vals)
    print(f"\n=== Property: {prop.upper()} | {n} molecules ===")

    aucs, accs = [], []
    for trial in range(N_TRIALS):
        # Reproduce the byte-identical split from run_experiments.py.
        split_generator = torch.Generator().manual_seed(42 + trial)
        train_len, val_len = int(0.8 * n), int(0.1 * n)
        test_len = n - train_len - val_len
        train_ds, val_ds, test_ds = random_split(
            np.arange(n),
            [train_len, val_len, test_len],
            generator=split_generator,
        )
        tr, te = np.array(train_ds), np.array(test_ds)
        # Same train-median labelling as the graph models: threshold on the
        # training split only, then applied to the held-out test set.
        train_median = float(np.median(vals[tr]))
        y = (vals > train_median).astype(float)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        scores = clf.decision_function(X[te])
        aucs.append(roc_auc_score(y[te], scores))
        accs.append(accuracy_score(y[te], clf.predict(X[te])))

    print(
        f"  logreg (composition, {X.shape[1] + 1} params): "
        f"AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} | "
        f"Acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}"
    )
    results.append({
        "Model": "logreg_composition",
        "Property": prop,
        "Params": X.shape[1] + 1,
        "Final_AUC_Mean": np.mean(aucs),
        "Final_AUC_Std_AcrossTrials": np.std(aucs),
        "Final_Acc_Mean": np.mean(accs),
        "Final_Acc_Std_AcrossTrials": np.std(accs),
        "Final_AUC_PerTrial": ";".join(f"{v:.6f}" for v in aucs),
        "Final_Acc_PerTrial": ";".join(f"{v:.6f}" for v in accs),
    })

pd.DataFrame(results).to_csv("trivial_baseline_metrics.csv", index=False)
print("\nSaved trivial_baseline_metrics.csv")
