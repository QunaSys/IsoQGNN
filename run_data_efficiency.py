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
Sample-complexity sweep (Fig. 3).

Sweeps the training-set size (50 -> 1500) and records test AUC on a fixed
500-molecule held-out set for both architectures on both tasks, over five
trials per size. For each size the reported epoch is chosen by validation AUC
(never the test set), applied identically to both architectures, on a shorter
40-epoch budget than the matched-capacity run for tractability across the
sweep. Writes data_efficiency_combined.csv and data_efficiency_combined.png.

QM9 is downloaded from the DeepChem mirror on first run.
"""
import io
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cgnn import IsoCGNN
from config import ATOM_TYPES, LEARNING_RATE, N_MAX
from dataset import MoleculeDataset
from preprocessing import preprocess_smiles
from q_trainer import QTrainer
from qgnn import IsoQGNN

# --- Configuration ---
PROPERTIES = ["gap", "mu"]
SIZES = [
    50, 100, 150, 200, 250, 300, 350, 400,
    500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
]
N_TRIALS = 5  # five trials per training size (19 sizes x 2 tasks x 2 models)
N_EPOCHS = 40  # shorter per-point budget than run_experiments.py, for tractability
BATCH_SIZE = 1
VAL_SIZE = 300  # disjoint validation set used for epoch selection

MODELS = {
    "quantum": IsoQGNN,
    "classical": IsoCGNN,
}

# --- Plotting palette and conventions ---
# QunaSys-inspired palette: deep navy + teal, chosen for consistent,
# colour-blind-safe figures throughout the paper.
QUNASYS_NAVY = "#0B2545"
QUNASYS_TEAL = "#1B998B"

MODEL_COLORS = {"quantum": QUNASYS_NAVY, "classical": QUNASYS_TEAL}
MODEL_LABELS = {"quantum": "Iso-QGNN", "classical": "Iso-CGNN"}

TASK_LINESTYLES = {"gap": "-", "mu": "--"}
TASK_LABELS = {"gap": "HOMO-LUMO Gap", "mu": "Dipole Moment"}

# --- Setup Data (Cached) ---
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
print("Downloading QM9...")
s = requests.get(URL).content
RAW_DF = pd.read_csv(io.StringIO(s.decode("utf-8")))


def get_full_dataset(target_col, sample_pool=3000, seed=42):
    """Return filtered (SMILES, continuous target) pairs. The median-split
    threshold is computed later on the training POOL only (excluding the fixed
    held-out test set), so the class boundary never sees the test targets."""
    df = RAW_DF.sample(n=sample_pool, random_state=seed).reset_index(drop=True)

    valid_smiles, valid_vals = [], []
    for smi, val in zip(df["smiles"], df[target_col].values):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if len(atoms) > N_MAX or not all(a in ATOM_TYPES for a in atoms):
            continue
        valid_smiles.append(smi)
        valid_vals.append(float(val))

    return valid_smiles, np.array(valid_vals)


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --- Main Efficiency Loop ---
final_results = {m: {} for m in MODELS}

for prop in PROPERTIES:
    print(f"\n=== Data Efficiency Experiment: {prop.upper()} ===")

    # Fixed test set (same molecules for both models)
    valid_smiles, valid_vals = get_full_dataset(prop)
    total_indices = list(range(len(valid_smiles)))
    train_pool_idx, test_idx = train_test_split(
        total_indices, test_size=500, random_state=42
    )
    # Binarise using the median of the TRAIN POOL only (the fixed 500-molecule
    # test set is excluded from the threshold); labels are then stable across
    # all training-set sizes, and the class boundary never sees the test set.
    pool_median = float(np.median(valid_vals[train_pool_idx]))
    labels = [1.0 if v > pool_median else 0.0 for v in valid_vals]
    full_dataset = MoleculeDataset(valid_smiles, labels, preprocess_smiles)
    test_ds = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: b[0]
    )
    print(f"Fixed Test Set Size: {len(test_ds)}")

    prop_stats = {
        m: {"size": [], "mean_auc": [], "std_auc": [], "mean_acc": [], "std_acc": []}
        for m in MODELS
    }

    for n_train in SIZES:
        print(f"  > Training Size: {n_train} samples")

        trial_results = {m: {"auc": [], "acc": []} for m in MODELS}

        for trial in range(N_TRIALS):
            # Sample training subset ONCE per trial; both models see identical data.
            np.random.seed(1000 * trial + n_train)
            subset_idx = np.random.choice(train_pool_idx, size=n_train, replace=False)
            train_ds = Subset(full_dataset, subset_idx)
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=lambda b: b[0],
            )

            # Disjoint validation set for epoch selection, drawn from the pool
            # (never overlapping the training subset or the fixed test set) via
            # a separate RNG stream so the training draw above is unperturbed.
            remaining = np.setdiff1d(np.array(train_pool_idx), subset_idx)
            val_rng = np.random.RandomState(7000 * trial + n_train)
            val_idx = val_rng.choice(
                remaining, size=min(VAL_SIZE, len(remaining)), replace=False
            )
            val_loader = DataLoader(
                Subset(full_dataset, list(val_idx)),
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=lambda b: b[0],
            )

            for model_name, ModelClass in MODELS.items():
                set_all_seeds(42 + trial)
                model = ModelClass(n_nodes=N_MAX)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                trainer = QTrainer(
                    model, train_loader, torch.nn.CrossEntropyLoss(), optimizer
                )

                # Select the epoch by validation AUC and report test there,
                # so the test set is never searched over; the choice is made
                # identically for both architectures.
                val_hist, test_auc_hist, test_acc_hist = [], [], []
                for ep in range(N_EPOCHS):
                    trainer.train_epoch()
                    _, _, _, v_auc = trainer.evaluate(val_loader, "Val")
                    _, acc, _, auc = trainer.evaluate(test_loader, "Test")
                    val_hist.append(v_auc)
                    test_auc_hist.append(auc)
                    test_acc_hist.append(acc)

                best_ep = int(np.argmax(val_hist))
                trial_results[model_name]["auc"].append(test_auc_hist[best_ep])
                trial_results[model_name]["acc"].append(test_acc_hist[best_ep])

        # Aggregate for this training size
        for model_name in MODELS:
            aucs = trial_results[model_name]["auc"]
            accs = trial_results[model_name]["acc"]
            prop_stats[model_name]["size"].append(n_train)
            prop_stats[model_name]["mean_auc"].append(np.mean(aucs))
            prop_stats[model_name]["std_auc"].append(np.std(aucs))
            prop_stats[model_name]["mean_acc"].append(np.mean(accs))
            prop_stats[model_name]["std_acc"].append(np.std(accs))
            print(
                f"    -> {model_name:10s}: "
                f"AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}"
            )

    for model_name in MODELS:
        final_results[model_name][prop] = prop_stats[model_name]

# --- PLOTTING ---
# Shared style with run_experiments.py: common width + font sizes so text
# renders at a uniform size at the shared column width; no in-figure title
# (the description lives in the LaTeX caption).
SHORT_TASK = {"gap": "Gap", "mu": "Dipole"}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})
plt.figure(figsize=(4.8, 4.8 * 0.66))

for prop in PROPERTIES:
    for model_name in MODELS:
        stats = final_results[model_name][prop]
        plt.errorbar(
            stats["size"],
            stats["mean_auc"],
            yerr=stats["std_auc"],
            fmt=f"{TASK_LINESTYLES[prop]}o",
            color=MODEL_COLORS[model_name],
            ecolor=MODEL_COLORS[model_name],
            alpha=0.9,
            capsize=3,
            lw=1.1,
            ms=4,
            label=f"{MODEL_LABELS[model_name]} · {SHORT_TASK[prop]}",
        )

plt.xlabel("Number of training molecules")
plt.ylabel("Test AUC (fixed 500-molecule set)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right", ncol=2)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("data_efficiency_combined.png", dpi=300, bbox_inches="tight")
print("\nSaved 'data_efficiency_combined.png'")

# --- SAVE DATA ---
all_rows = []
for model_name in MODELS:
    for prop in PROPERTIES:
        stats = final_results[model_name][prop]
        for i, size in enumerate(stats["size"]):
            all_rows.append(
                {
                    "Model": model_name,
                    "Property": prop,
                    "Size": size,
                    "Mean_AUC": stats["mean_auc"][i],
                    "Std_AUC": stats["std_auc"][i],
                    "Mean_Acc": stats["mean_acc"][i],
                    "Std_Acc": stats["std_acc"][i],
                }
            )

pd.DataFrame(all_rows).to_csv("data_efficiency_combined.csv", index=False)
print("Saved 'data_efficiency_combined.csv'")

