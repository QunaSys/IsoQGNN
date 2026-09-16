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
Training loop shared by Iso-QGNN and Iso-CGNN.

QTrainer drives either architecture unchanged (both expose ``theta_atom``,
``theta_bond``, ``gnn_params`` and the same forward signature), which is what
makes the matched comparison a fair one. Alongside the loss it accumulates the
mean gradient norm of each parameter group per epoch -- the quantity plotted
in the gradient-stability figure and used to check for barren plateaus.
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class QTrainer:
    """Minimal trainer with per-parameter-group gradient-norm tracking.

    Args:
        model: an Iso-QGNN or Iso-CGNN instance.
        dataloader: yields one molecule dict per step (batch size one).
        criterion: loss (cross-entropy in the paper).
        optimizer: e.g. Adam.
        accumulation_steps: optimiser steps every this many samples
            (1 = update per molecule, as used for the reported results).
    """

    def __init__(
        self, model, dataloader, criterion, optimizer, accumulation_steps=1
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps

        # Trackers
        self.grad_atom_acc = 0.0
        self.grad_bond_acc = 0.0
        self.grad_gnn_acc = 0.0
        self.num_batches = 0

    def train_epoch(self) -> float:
        self.model.train()

        # Reset trackers
        self.grad_atom_acc = 0.0
        self.grad_bond_acc = 0.0
        self.grad_gnn_acc = 0.0
        self.num_batches = 0

        total_loss = 0.0
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.dataloader):
            atoms = batch["atoms"]
            bonds = batch["bonds"]
            label = batch["label"]

            # 1. Forward Pass
            logits = self.model(atoms, bonds)
            target = torch.tensor([label], dtype=torch.long, device=logits.device)
            loss = self.criterion(logits, target)

            # 2. Scale loss for accumulation
            loss = loss / self.accumulation_steps
            loss.backward()

            # Scale loss back up for accurate reporting
            loss_item = loss.item() * self.accumulation_steps
            total_loss += loss_item

            # 3. Conditional Update (The "Virtual Batch" Step)
            if (i + 1) % self.accumulation_steps == 0:
                self.grad_atom_acc += self.grad_norm(self.model.theta_atom)
                self.grad_bond_acc += self.grad_norm(self.model.theta_bond)
                self.grad_gnn_acc += self.grad_norm(self.model.gnn_params)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.num_batches += 1

        # 4. Handle Remainder (Final partial batch)
        if len(self.dataloader) % self.accumulation_steps != 0:
            self.grad_atom_acc += self.grad_norm(self.model.theta_atom)
            self.grad_bond_acc += self.grad_norm(self.model.theta_bond)
            self.grad_gnn_acc += self.grad_norm(self.model.gnn_params)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.num_batches += 1

        return total_loss / len(self.dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader=None, dataset_name="Validation"):
        self.model.eval()
        loader = dataloader if dataloader is not None else self.dataloader

        y_true, y_pred, y_probs = [], [], []
        running_loss = 0.0

        for batch in loader:
            atoms, bonds, label = batch["atoms"], batch["bonds"], batch["label"]
            logits = self.model(atoms, bonds)

            # Device safety
            target = torch.tensor([label], dtype=torch.long, device=logits.device)
            loss = self.criterion(logits, target)

            running_loss += loss.item()
            pred = torch.argmax(logits, dim=1).item()
            prob = F.softmax(logits, dim=1)[:, 1].item()

            y_true.append(label)
            y_pred.append(pred)
            y_probs.append(prob)

        avg_loss = running_loss / len(loader)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Calculate AUC (Handle edge case if only 1 class exists in batch)
        try:
            auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            auc = 0.5

        # Print metrics
        print(
            f"{dataset_name} Metrics: Loss: {avg_loss:.4f} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}"
        )

        # Gradient & Parameter Printing
        if dataset_name == "Train":
            print(
                f"Parameters: "
                f"||θ_atom|| = {self.param_norm(self.model.theta_atom):.4f} | "
                f"||θ_bond|| = {self.param_norm(self.model.theta_bond):.4f} | "
                f"||θ_gnn|| = {self.param_norm(self.model.gnn_params):.4f}"
            )

            denom = self.num_batches if self.num_batches > 0 else 1
            print(
                f"Gradients:  "
                f"∇θ_atom = {self.grad_atom_acc / denom:.3e} | "
                f"∇θ_bond = {self.grad_bond_acc / denom:.3e} | "
                f"∇θ_gnn = {self.grad_gnn_acc / denom:.3e}"
            )

        return avg_loss, acc, f1, auc

    def param_norm(self, param):
        return torch.norm(param.detach()).item()

    def grad_norm(self, param):
        if param.grad is None:
            return 0.0
        return torch.norm(param.grad.detach()).item()
