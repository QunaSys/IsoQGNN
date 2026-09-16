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
Matched-capacity comparison (Table III and Figs. 2, 4, 5).

Trains Iso-QGNN and Iso-CGNN on byte-identical splits over ten seeds and 200
epochs, on both the HOMO-LUMO gap and dipole classification tasks, and writes:
  - experiment_summary_metrics.csv  (Table III: mean +/- std AUC/accuracy)
  - experiment_full_curves.csv      (per-epoch trial-averaged curves)
  - experiment_confusion_matrices.png, experiment_gradient_stability.png,
    final_bonds_errorbars.png, and supplementary loss/metric plots.

QM9 is downloaded from the DeepChem mirror on first run. Reported metrics are
final-epoch values at convergence; see the paper's Training Protocol.
"""
import io
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from rdkit import Chem
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

from cgnn import IsoCGNN
from config import ATOM_TYPES, LEARNING_RATE, N_MAX
from dataset import MoleculeDataset
from preprocessing import preprocess_smiles
from q_trainer import QTrainer
from qgnn import IsoQGNN

# --- Configuration ---
# The variational quantum models need training to convergence: with a shorter
# budget some seeds remain under-converged, most visibly on the gap task.
# N_TRIALS=10 gives stable cross-trial statistics given that seed sensitivity.
N_TRIALS = 10
N_EPOCHS = 200
BATCH_SIZE = 1

# Models to compare. Order matters only for plotting conventions.
MODELS = {
    "quantum": IsoQGNN,
    "classical": IsoCGNN,
}

# --- Plotting palette and conventions ---
# QunaSys-inspired palette: deep navy + teal, chosen for consistent,
# colour-blind-safe figures throughout the paper.
QUNASYS_NAVY = "#0B2545"
QUNASYS_TEAL = "#1B998B"
QUNASYS_INK = "#1A1A2E"  # near-black for headings/labels (softer than pure black)

# Model is the colour-coded variable (matches matched-comparison framing).
MODEL_COLORS = {"quantum": QUNASYS_NAVY, "classical": QUNASYS_TEAL}
MODEL_LABELS = {"quantum": "Iso-QGNN", "classical": "Iso-CGNN"}
MODEL_HATCHES = {"quantum": "", "classical": "//"}

# Task is differentiated by linestyle within colour.
TASK_LINESTYLES = {"gap": "-", "mu": "--"}
TASK_LABELS = {"gap": "HOMO-LUMO Gap", "mu": "Dipole Moment"}

# Per-parameter-group palette for the gradient-stability figure.
# Three desaturated tones drawn from the same family as the brand colours,
# so the gradient figure stays visually consistent with the rest of the paper.
GRAD_COLORS = {
    "atom": QUNASYS_NAVY,
    "bond": QUNASYS_TEAL,
    "gnn": "#6B7A8F",  # neutral slate
}

# --- Setup Data (Optimized) ---
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
print("Downloading QM9...")
s = requests.get(URL).content
RAW_DF = pd.read_csv(io.StringIO(s.decode("utf-8")))


def get_dataset(target_col, sample_size=1500, seed=42):
    """Return filtered (SMILES, continuous target) pairs. Binary labels are NOT
    assigned here: the median-split threshold is computed per trial on the
    TRAIN split only (see the main loop), so the class boundary never sees the
    held-out validation/test targets."""
    df = RAW_DF.sample(n=sample_size, random_state=seed).reset_index(drop=True)

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
    """Reset all RNG state so model init is reproducible across models in a trial."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --- Main Experiment Loop ---
# Results keyed as: final_results[model_name][property]
final_results = {m: {} for m in MODELS}

for prop in ["gap", "mu"]:
    valid_smiles, valid_vals = get_dataset(prop)
    total_len = len(valid_smiles)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    # Per-model accumulators for this property
    per_model = {
        m: {
            "trial_bonds": [],
            "trial_cms": [],
            "trial_times": [],
            "curves": {
                "train_loss": [],
                "test_loss": [],
                "test_auc": [],
                "test_acc": [],
                "grad_atom": [],
                "grad_bond": [],
                "grad_gnn": [],
            },
            "param_count": 0,
        }
        for m in MODELS
    }

    for trial in range(N_TRIALS):
        print(f"\n=== Property: {prop.upper()} | Trial {trial + 1}/{N_TRIALS} ===")

        # 1. Build splits ONCE per trial (identical partition for both models)
        #    from a seeded permutation, matching torch.random_split so the
        #    partition is reproducible and shared with the non-graph control.
        split_generator = torch.Generator().manual_seed(42 + trial)
        perm = torch.randperm(total_len, generator=split_generator).tolist()
        train_idx = perm[:train_len]
        val_idx = perm[train_len:train_len + val_len]
        test_idx = perm[train_len + val_len:]
        # Binarise using the TRAIN-split median ONLY, then apply that threshold
        # to val/test -- the class boundary never sees held-out targets.
        train_median = float(np.median(valid_vals[train_idx]))
        labels = [1.0 if v > train_median else 0.0 for v in valid_vals]
        dataset = MoleculeDataset(valid_smiles, labels, preprocess_smiles)
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: b[0]
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: b[0]
        )

        # 2. Train each model on identical splits with identical init seeds.
        for model_name, ModelClass in MODELS.items():
            print(f"  > Training {model_name}...")
            start_t = time.time()

            # Reset RNG so both models are initialized from the same state.
            set_all_seeds(42 + trial)
            model = ModelClass(n_nodes=N_MAX)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trial == 0:
                print(f"    [Info] {model_name} Parameter Count: {param_count}")
                per_model[model_name]["param_count"] = param_count

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            trainer = QTrainer(
                model, train_loader, torch.nn.CrossEntropyLoss(), optimizer
            )

            h_train_loss, h_test_loss, h_auc, h_acc = [], [], [], []
            h_grad_atom, h_grad_bond, h_grad_gnn = [], [], []

            for ep in range(N_EPOCHS):
                t_loss = trainer.train_epoch()
                v_loss, acc, f1, auc = trainer.evaluate(test_loader, "Test")

                h_train_loss.append(t_loss)
                h_test_loss.append(v_loss)
                h_auc.append(auc)
                h_acc.append(acc)

                denom = trainer.num_batches if trainer.num_batches > 0 else 1
                h_grad_atom.append((trainer.grad_atom_acc / denom) + 1e-9)
                h_grad_bond.append((trainer.grad_bond_acc / denom) + 1e-9)
                h_grad_gnn.append((trainer.grad_gnn_acc / denom) + 1e-9)

            elapsed = time.time() - start_t
            per_model[model_name]["trial_times"].append(elapsed)

            per_model[model_name]["curves"]["train_loss"].append(h_train_loss)
            per_model[model_name]["curves"]["test_loss"].append(h_test_loss)
            per_model[model_name]["curves"]["test_auc"].append(h_auc)
            per_model[model_name]["curves"]["test_acc"].append(h_acc)
            per_model[model_name]["curves"]["grad_atom"].append(h_grad_atom)
            per_model[model_name]["curves"]["grad_bond"].append(h_grad_bond)
            per_model[model_name]["curves"]["grad_gnn"].append(h_grad_gnn)

            # Bond parameters: average over layers (a single layer here, so
            # this is the raw four-element vector).
            with torch.no_grad():
                bonds = model.theta_bond.mean(dim=0).cpu().numpy()
                per_model[model_name]["trial_bonds"].append(bonds)

            # Confusion matrix
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    logits = model(batch["atoms"], batch["bonds"])
                    preds = torch.argmax(logits, dim=1).item()
                    y_true.append(batch["label"])
                    y_pred.append(preds)
            per_model[model_name]["trial_cms"].append(confusion_matrix(y_true, y_pred))

    # --- Aggregate per model ---
    for model_name in MODELS:
        pm = per_model[model_name]
        avg_curves = {k: np.mean(v, axis=0) for k, v in pm["curves"].items()}
        std_curves = {k: np.std(v, axis=0) for k, v in pm["curves"].items()}

        # Per-trial final-epoch values; needed for cross-trial SD in the headline table.
        final_per_trial = {
            k: [trial_curve[-1] for trial_curve in pm["curves"][k]]
            for k in pm["curves"]
        }

        final_results[model_name][prop] = {
            "curves": avg_curves,
            "curves_std": std_curves,
            "final_per_trial": final_per_trial,
            "bonds_mean": np.mean(pm["trial_bonds"], axis=0),
            "bonds_std": np.std(pm["trial_bonds"], axis=0),
            "cm": np.sum(pm["trial_cms"], axis=0),
            "param_count": pm["param_count"],
            "avg_time": np.mean(pm["trial_times"]),
        }

# --- PLOTTING SECTION ---
# Shared figure style: a common width (FIG_W) and font sizes across every paper
# figure, so that when each is scaled to the same column width the text renders
# at a uniform size. Figures carry no in-figure titles -- the descriptions live
# in the LaTeX captions, per standard practice.
FIG_W = 4.8  # ~single-column display width -> ~8pt text at 0.48\textwidth
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

# Learned bond parameters (paper Fig. 5) - 2x1 stacked (Gap top, Dipole bottom) with
# independent y-axes (the two tasks have very different weight scales).
bond_labels = ["Single", "Double", "Triple", "Aromatic"]
x = np.arange(len(bond_labels))
width = 0.38

fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_W * 1.02))
for ax, prop in zip(axes, ["gap", "mu"]):
    q = final_results["quantum"][prop]
    c = final_results["classical"][prop]
    ax.bar(
        x - width / 2,
        q["bonds_mean"],
        width,
        yerr=q["bonds_std"],
        capsize=4,
        label=MODEL_LABELS["quantum"],
        color=MODEL_COLORS["quantum"],
        edgecolor="white",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    )
    ax.bar(
        x + width / 2,
        c["bonds_mean"],
        width,
        yerr=c["bonds_std"],
        capsize=4,
        label=MODEL_LABELS["classical"],
        color=MODEL_COLORS["classical"],
        edgecolor="white",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "#555"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bond_labels)
    ax.axhline(0, color="#999", linewidth=1)
    ax.set_ylabel("Learned bond weight")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.savefig("final_bonds_errorbars.png", dpi=300, bbox_inches="tight")
print("\nSaved 'final_bonds_errorbars.png'")


# Learning curves (AUC + Accuracy); supplementary, not in the paper
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for prop in ["gap", "mu"]:
    for model_name in MODELS:
        res = final_results[model_name][prop]
        ax1.plot(
            res["curves"]["test_auc"],
            linestyle=TASK_LINESTYLES[prop],
            color=MODEL_COLORS[model_name],
            label=f"{MODEL_LABELS[model_name]} ({TASK_LABELS[prop]})",
        )
        ax2.plot(
            res["curves"]["test_acc"],
            linestyle=TASK_LINESTYLES[prop],
            color=MODEL_COLORS[model_name],
            label=f"{MODEL_LABELS[model_name]} ({TASK_LABELS[prop]})",
        )

ax1.axhline(0.5, ls=":", color="gray", label="Random")
ax1.set_title(f"Test AUC (Avg of {N_TRIALS} Trials)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("AUC")
ax1.legend(fontsize=8)
ax1.grid(True)

ax2.axhline(0.5, ls=":", color="gray", label="Random")
ax2.set_title(f"Test Accuracy (Avg of {N_TRIALS} Trials)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend(fontsize=8)
ax2.grid(True)
plt.tight_layout()
plt.savefig("experiment_metrics_mean.png", dpi=300)
print("Saved 'experiment_metrics_mean.png'")


# Loss curves; supplementary, not in the paper
plt.figure(figsize=(10, 6))
for prop in ["gap", "mu"]:
    for model_name in MODELS:
        res = final_results[model_name][prop]
        base_label = f"{MODEL_LABELS[model_name]} ({TASK_LABELS[prop]})"
        plt.plot(
            res["curves"]["train_loss"],
            linestyle=TASK_LINESTYLES[prop],
            color=MODEL_COLORS[model_name],
            alpha=0.45,
            label=f"{base_label} Train",
        )
        plt.plot(
            res["curves"]["test_loss"],
            linestyle=TASK_LINESTYLES[prop],
            color=MODEL_COLORS[model_name],
            alpha=1.0,
            label=f"{base_label} Test",
        )
plt.title(f"Loss Convergence (Avg of {N_TRIALS} Trials)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("experiment_loss_curves_mean.png", dpi=300)
print("Saved 'experiment_loss_curves_mean.png'")


# Confusion matrices (paper Fig. 2) - 2x2 (rows = model, cols = task). No per-panel
# editorial titles: the task is a column header and the model a row label, so
# the grid is identified without a title (layout also stated in the caption).
fig4, axes4 = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 0.92))
for i, model_name in enumerate(MODELS):
    for j, prop in enumerate(["gap", "mu"]):
        cm = final_results[model_name][prop]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=axes4[i, j], annot_kws={"size": 9})
        if i == 0:
            axes4[i, j].set_title(TASK_LABELS[prop], color=QUNASYS_INK, pad=8)
        axes4[i, j].set_xlabel("Predicted label" if i == 1 else "")
        axes4[i, j].set_ylabel("True label" if j == 0 else "")
for i, model_name in enumerate(MODELS):
    fig4.text(-0.02, 0.74 - i * 0.46, MODEL_LABELS[model_name], rotation=90,
              va="center", ha="center", fontsize=15, fontweight="bold",
              color=QUNASYS_INK)
plt.tight_layout()
plt.savefig("experiment_confusion_matrices.png", dpi=300, bbox_inches="tight")
print("Saved 'experiment_confusion_matrices.png'")


# Gradient stability (paper Fig. 4) - 2x1 stacked (Quantum top, Classical bottom),
# Gap task only. Vertically stacked with shared x and y axes: both substrates
# occupy the same 1e-1..1e0 band, so a shared scale lets the reader read the
# "same band, different dynamics" contrast top-to-bottom (quantum climbs from a
# small init to a plateau; classical decays from a larger start).
fig5, axes5 = plt.subplots(
    2, 1, figsize=(FIG_W, FIG_W * 1.12), sharex=True, sharey=True
)
for ax, model_name in zip(axes5, MODELS):
    c = final_results[model_name]["gap"]["curves"]
    ax.plot(
        c["grad_atom"],
        label=r"$\nabla \theta_{atom}$",
        color=GRAD_COLORS["atom"],
        alpha=0.9,
    )
    ax.plot(
        c["grad_bond"],
        label=r"$\nabla \theta_{bond}$",
        color=GRAD_COLORS["bond"],
        alpha=0.9,
    )
    ax.plot(
        c["grad_gnn"],
        label=r"$\nabla \theta_{gnn}$",
        color=GRAD_COLORS["gnn"],
        alpha=0.9,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Average gradient norm")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes5[0].legend(loc="lower right", ncol=3)
axes5[-1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("experiment_gradient_stability.png", dpi=300, bbox_inches="tight")
print("Saved 'experiment_gradient_stability.png'")


# --- SAVE RAW DATA TO CSV ---
print("\n=== SAVING RAW DATA ===")

# 1. Curve data (long format, mean across trials, keyed by Model + Property)
all_curve_data = []
for model_name in MODELS:
    for prop in ["gap", "mu"]:
        c = final_results[model_name][prop]["curves"]
        for i, ep in enumerate(range(1, N_EPOCHS + 1)):
            all_curve_data.append(
                {
                    "Model": model_name,
                    "Property": prop,
                    "Epoch": ep,
                    "Train_Loss": c["train_loss"][i],
                    "Test_Loss": c["test_loss"][i],
                    "Test_AUC": c["test_auc"][i],
                    "Test_Acc": c["test_acc"][i],
                    "Grad_Atom": c["grad_atom"][i],
                    "Grad_Bond": c["grad_bond"][i],
                    "Grad_GNN": c["grad_gnn"][i],
                }
            )
pd.DataFrame(all_curve_data).to_csv("experiment_full_curves.csv", index=False)
print("Saved 'experiment_full_curves.csv'")

# 2. Summary metrics — cross-trial SDs and per-trial values
summary_data = []
for model_name in MODELS:
    for prop in ["gap", "mu"]:
        res = final_results[model_name][prop]
        per_trial_auc = res["final_per_trial"]["test_auc"]
        per_trial_acc = res["final_per_trial"]["test_acc"]
        row = {
            "Model": model_name,
            "Property": prop,
            "Params": res["param_count"],
            "Avg_Time_Sec": res["avg_time"],
            "Final_AUC_Mean": float(np.mean(per_trial_auc)),
            "Final_AUC_Std_AcrossTrials": float(np.std(per_trial_auc)),
            "Final_Acc_Mean": float(np.mean(per_trial_acc)),
            "Final_Acc_Std_AcrossTrials": float(np.std(per_trial_acc)),
            "Final_AUC_PerTrial": ";".join(f"{v:.6f}" for v in per_trial_auc),
            "Final_Acc_PerTrial": ";".join(f"{v:.6f}" for v in per_trial_acc),
        }
        for i, bl in enumerate(bond_labels):
            row[f"Bond_{bl}_Mean"] = res["bonds_mean"][i]
            row[f"Bond_{bl}_Std"] = res["bonds_std"][i]
        summary_data.append(row)
pd.DataFrame(summary_data).to_csv("experiment_summary_metrics.csv", index=False)
print("Saved 'experiment_summary_metrics.csv'")

print("\n=== FINAL RESULTS SUMMARY ===")
for model_name in MODELS:
    for prop in ["gap", "mu"]:
        res = final_results[model_name][prop]
        per_trial_auc = res["final_per_trial"]["test_auc"]
        per_trial_acc = res["final_per_trial"]["test_acc"]
        auc_mean, auc_std = np.mean(per_trial_auc), np.std(per_trial_auc)
        acc_mean, acc_std = np.mean(per_trial_acc), np.std(per_trial_acc)
        print(
            f"Model: {model_name:10s} | Prop: {prop.upper()} | "
            f"Params: {res['param_count']} | "
            f"AUC: {auc_mean:.3f} +/- {auc_std:.3f} | "
            f"Acc: {acc_mean:.3f} +/- {acc_std:.3f} | "
            f"Time/Trial: {res['avg_time']:.1f}s"
        )

