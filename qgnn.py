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
Iso-QGNN: the topology-aligned variational quantum circuit.

The circuit mirrors the molecular bond graph. Each heavy atom is assigned one
qubit; bonded atoms are entangled. A forward pass proceeds in three stages
(see the Method section of the paper):

  1. Atom embedding      -- R_Y(theta_atom) per atom type, on |0>.
  2. Isomorphic          -- for each bond, CZ followed by symmetric
     entanglement          R_Z(theta_bond) rotations, with parameters shared
                            across all bonds of a given type.
  3. Variational         -- R_X, R_Y rotations on every qubit (the node-update
     rotations             analogue).

The register is read out as the per-qubit magnetisation <Z_i>, which a small
classical head (LayerNorm + Linear) maps to class logits. Parameters are
shared by atom type and bond type, giving 64 trainable parameters at a single
layer (n_layers=1) -- the matched capacity used throughout the paper.
"""
import math

import pennylane as qml
import torch
import torch.nn as nn

from config import ATOM_TYPES, N_CLASSES, N_LAYERS, QUBITS_PER_NODE
from preprocessing import PADDING_ATOM

"""
BOND_TYPES_DEFAULT (Dict[int, int]):
    A mapping from RDKit bond types (integers) to quantum circuit
    parameter indices.
    - Keys: RDKit BondType (1=Single, 2=Double, 3=Triple, 4=Aromatic)).
    - Values: Index of the trainable parameter theta_bond to use for
      that connection.
"""
BOND_TYPES_DEFAULT = {1: 0, 2: 1, 3: 2, 4: 3}


class IsoQGNN(nn.Module):
    """Topology-aligned variational quantum circuit (Iso-QGNN).

    Args:
        n_nodes: register width (max heavy atoms, N_MAX); one qubit per node.
        qubits_per_node: qubits per atom (1 in the paper).
        n_layers: repetitions of the entangle+rotate block (1 in the paper).
        n_classes: output classes (2 for the binary tasks).

    Forward signature (shared with IsoCGNN so the same trainer drives
    both): ``model(atoms: list[str], bonds: list[tuple]) -> logits [1, n_classes]``,
    where ``atoms`` is a length-n_nodes list of symbols (PADDING_ATOM for
    padding) and ``bonds`` is a list of ``(i, j, bond_type)`` triples.
    """

    def __init__(
        self,
        n_nodes,
        qubits_per_node=QUBITS_PER_NODE,
        n_layers=N_LAYERS,
        n_classes=N_CLASSES,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.qubits_per_node = qubits_per_node
        self.n_qubits = n_nodes * qubits_per_node
        self.n_layers = n_layers

        # 1. Define Device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # 2. Define Parameters
        # Note: Initialize atom embeddings with wider variance (0 to 2pi)
        # This breaks symmetry so 'C' and 'O' look different from epoch 0.
        # Uses Learned Embedding.
        self.theta_atom = nn.Parameter(
            torch.rand(len(ATOM_TYPES), qubits_per_node) * 2 * math.pi
        )

        # Note: Keep bonds/GNN params small to start near Identity
        self.theta_bond = nn.Parameter(
            0.01 * torch.randn(n_layers, len(BOND_TYPES_DEFAULT))
        )
        self.gnn_params = nn.Parameter(
            0.01 * torch.randn(n_layers, self.n_qubits, 2)
        )

        # 3. Create QNode
        self.qnode = qml.QNode(self._circuit_impl, self.dev, interface="torch")

        # 4. Classical Head
        self.layer_norm = nn.LayerNorm(self.n_qubits)
        self.classifier = nn.Linear(self.n_qubits, n_classes)

    def _encode_atoms(self, atoms: list[str], theta_atom: torch.Tensor):
        """Stage 1: state preparation. Apply R_Y(theta_atom) to each
        non-padding qubit, with a per-atom-type angle. Padding atoms are left
        in |0> and receive no gate."""
        for node_idx, atom in enumerate(atoms):
            if atom == PADDING_ATOM:
                continue
            atom_id = ATOM_TYPES.index(atom)
            for q in range(self.qubits_per_node):
                wire = node_idx * self.qubits_per_node + q
                qml.RY(theta_atom[atom_id, q], wires=wire)

    def _quantum_layer(
        self,
        atoms: list[str],
        bonds: list[tuple],
        theta_bond: torch.Tensor,
        gnn_params: torch.Tensor,
        layer_idx: int,
    ):
        """One layer: stage 2 (isomorphic entanglement) then stage 3
        (variational rotations). For each bond, apply CZ + symmetric
        R_Z(theta_bond) with a per-bond-type angle; then apply R_X, R_Y on
        every non-padding qubit. Bonds touching a padding atom are skipped."""
        # Bond entanglement
        for i, j, bond_type in bonds:
            if atoms[i] == PADDING_ATOM or atoms[j] == PADDING_ATOM:
                continue

            bond_id = BOND_TYPES_DEFAULT.get(bond_type, 0)
            node_i_qubits = range(
                i * self.qubits_per_node, (i + 1) * self.qubits_per_node
            )
            node_j_qubits = range(
                j * self.qubits_per_node, (j + 1) * self.qubits_per_node
            )

            theta = theta_bond[layer_idx, bond_id]
            for qi, qj in zip(node_i_qubits, node_j_qubits):
                qml.CZ(wires=[qi, qj])
                qml.RZ(theta, wires=qi)
                qml.RZ(theta, wires=qj)

        # Node update rotations (skip padding qubits)
        for node_idx, atom in enumerate(atoms):
            if atom == PADDING_ATOM:
                continue
            for q in range(self.qubits_per_node):
                wire = node_idx * self.qubits_per_node + q
                qml.RX(gnn_params[layer_idx, wire, 0], wires=wire)
                qml.RY(gnn_params[layer_idx, wire, 1], wires=wire)

    def _circuit_impl(self, atoms, bonds, theta_atom, theta_bond, gnn_params):
        """The QNode body: encode atoms once, then apply n_layers entangle+
        rotate blocks, and measure <Z_i> on every qubit. Returns one
        expectation per qubit (the molecular feature vector)."""
        # Atom encoding applied once as initial state preparation
        self._encode_atoms(atoms, theta_atom)

        for layer_idx in range(self.n_layers):
            self._quantum_layer(
                atoms, bonds, theta_bond, gnn_params, layer_idx
            )

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, atoms, bonds):
        """Run the circuit for one molecule and map the <Z_i> readout through
        the classical head (LayerNorm + Linear) to class logits [1, n_classes]."""
        q_out = self.qnode(
            atoms, bonds, self.theta_atom, self.theta_bond, self.gnn_params
        )

        graph_embedding = torch.stack(q_out).float().unsqueeze(0)

        norm_embedding = self.layer_norm(graph_embedding)
        logits = self.classifier(norm_embedding)

        return logits
