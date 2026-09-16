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

# cgnn.py
"""
Classical Isomorphic GNN - Parameter-Matched Baseline for Iso-QGNN
===================================================================

Mirrors the Iso-QGNN architecture one-for-one, replacing only the entangling
primitive (CZ) with a classical message-passing update. Every other structural
choice - parameter sharing, per-layer counts, readout head - is identical.

  Quantum primitive              ->  Classical analog
  ------------------------------     ----------------------------------
  RY(theta_atom) on |0>          ->  cos(theta) per atom type (mirrors <Z>)
  CZ + RZ(theta_bond) on bond    ->  shared bond-type weighted message
  RX, RY per qubit per layer     ->  scale + shift per node per layer
  <Z_i> measurement              ->  scalar node state (tanh-bounded)
  LayerNorm + Linear (identical) ->  LayerNorm + Linear (identical)

Attribute names match IsoQGNN (theta_atom, theta_bond, gnn_params) so
QTrainer's gradient-norm tracking works without modification, so the two
models are interchangeable wherever QTrainer is used (one import swap).

Total parameter count with the default config (N_MAX=9, n_layers=1,
4 atom types, 4 bond types, 2 classes) is exactly 64, matching IsoQGNN.
"""

import math

import torch
import torch.nn as nn

from config import ATOM_TYPES, N_CLASSES, N_LAYERS
from preprocessing import PADDING_ATOM
from qgnn import BOND_TYPES_DEFAULT


class IsoCGNN(nn.Module):
    """
    Classical twin of IsoQGNN.

    Forward signature matches exactly:
        model(atoms: list[str], bonds: list[tuple]) -> logits [1, n_classes]
    """

    def __init__(
        self,
        n_nodes,
        n_layers=N_LAYERS,
        n_classes=N_CLASSES,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers

        # --- Parameters (names mirror IsoQGNN for QTrainer compatibility) ---

        # Atom embedding: 1 scalar per atom type.
        # Initialized uniformly in [0, 2pi] to break symmetry, matching the
        # quantum model where theta_atom is initialized over a full rotation.
        self.theta_atom = nn.Parameter(
            torch.rand(len(ATOM_TYPES)) * 2 * math.pi
        )

        # Bond entanglement: 1 scalar per (bond type, layer).
        # Shape and initialization mirror the quantum theta_bond exactly.
        self.theta_bond = nn.Parameter(
            0.01 * torch.randn(n_layers, len(BOND_TYPES_DEFAULT))
        )

        # Node-wise update: (scale, shift) per node per layer.
        # Same shape as the quantum gnn_params (RX, RY rotations per qubit).
        self.gnn_params = nn.Parameter(
            0.01 * torch.randn(n_layers, n_nodes, 2)
        )

        # --- Classical head (identical module to IsoQGNN) ---
        self.layer_norm = nn.LayerNorm(n_nodes)
        self.classifier = nn.Linear(n_nodes, n_classes)

    def forward(self, atoms, bonds):
        device = self.theta_atom.device

        # --- Node initialization: analog of RY|0> -> <Z> = cos(theta) ---
        # Padding nodes start at 0.0 (equivalent to an unrotated |0> which
        # gives <Z> = 1, but we use 0 here so the zero-init baseline has no
        # contribution; the readout LayerNorm handles the offset).
        h_components = []
        for atom in atoms:
            if atom == PADDING_ATOM:
                h_components.append(torch.zeros((), device=device))
            else:
                atom_id = ATOM_TYPES.index(atom)
                h_components.append(torch.cos(self.theta_atom[atom_id]))
        h = torch.stack(h_components)  # [n_nodes]

        # --- Layered update ---
        for l in range(self.n_layers):
            # Bond-type-shared message passing (analog of CZ + RZ on bonded pairs)
            msg_components = [torch.zeros((), device=device) for _ in range(self.n_nodes)]
            for bi, bj, bond_type in bonds:
                # Match quantum model: skip bonds involving padding nodes
                if atoms[bi] == PADDING_ATOM or atoms[bj] == PADDING_ATOM:
                    continue
                bond_id = BOND_TYPES_DEFAULT.get(bond_type, 0)
                w = self.theta_bond[l, bond_id]
                # Symmetric neighbour contribution
                msg_components[bi] = msg_components[bi] + w * h[bj]
                msg_components[bj] = msg_components[bj] + w * h[bi]
            messages = torch.stack(msg_components)
            h = h + messages

            # Per-node update (analog of RX, RY -> Z measurement). tanh keeps
            # the state bounded like a Z expectation. Applied to all nodes
            # including padding: a padding node therefore reads tanh(shift), a
            # learned near-zero constant, whereas the quantum code skips padding
            # qubits so their <Z> stays exactly +1. Both are constant offsets
            # that carry no information beyond the heavy-atom count.
            scale = self.gnn_params[l, :, 0]
            shift = self.gnn_params[l, :, 1]
            h = torch.tanh(scale * h + shift)

        # --- Readout (identical to IsoQGNN) ---
        graph_embedding = h.unsqueeze(0)  # [1, n_nodes]
        normed = self.layer_norm(graph_embedding)
        logits = self.classifier(normed)
        return logits


def count_parameters(model: nn.Module) -> dict:
    """Breakdown-by-name parameter counter - useful for verifying matched capacity."""
    counts = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    counts["TOTAL"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    # Sanity check: parameter count should match IsoQGNN exactly.
    from qgnn import IsoQGNN

    q = IsoQGNN(n_nodes=9)
    c = IsoCGNN(n_nodes=9)

    print("IsoQGNN parameters:")
    for k, v in count_parameters(q).items():
        print(f"  {k:30s} {v}")
    print("\nIsoCGNN parameters:")
    for k, v in count_parameters(c).items():
        print(f"  {k:30s} {v}")

