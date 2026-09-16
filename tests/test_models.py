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

"""Model invariants: the matched 64-parameter capacity and circuit inventory.

These guard the paper's central "matched capacity" claim (Iso-QGNN and
Iso-CGNN both have exactly 64 trainable parameters at the default single-layer
config) and the per-layer gate count stated in the scaling analysis.
"""
import pennylane as qml
import pytest

from cgnn import IsoCGNN
from preprocessing import preprocess_smiles
from qgnn import IsoQGNN

N_NODES = 9


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_matched_64_parameters():
    """Both architectures have exactly 64 params at the default (L=1) config."""
    assert n_params(IsoQGNN(n_nodes=N_NODES)) == 64
    assert n_params(IsoCGNN(n_nodes=N_NODES)) == 64


def test_parameter_count_scales_with_layers():
    """Documents the L-dependence: L=2 -> 86, L=3 -> 108 (see Table II text)."""
    assert n_params(IsoQGNN(n_nodes=N_NODES, n_layers=2)) == 86
    assert n_params(IsoQGNN(n_nodes=N_NODES, n_layers=3)) == 108


@pytest.mark.parametrize(
    "atoms,bonds,n,e",
    [
        (["N", "C", "O"], [(0, 1, 1), (1, 2, 2)], 3, 2),
        (
            ["C", "C", "C", "C", "N", "O"],
            [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (0, 5, 4)],
            6,
            6,
        ),
    ],
)
def test_gate_count_at_one_layer_is_3n_plus_3e(atoms, bonds, n, e):
    """At the single layer used throughout, the circuit has 3N + 3|E| gates:
    one RY encoding plus two variational rotations per atom, and a CZ with two
    RZ rotations per bond. The atom encoding is applied once, before the layer,
    so the general total is N + L(2N + 3|E|)."""
    m = IsoQGNN(n_nodes=len(atoms))
    specs = qml.specs(m.qnode)(
        atoms, bonds, m.theta_atom, m.theta_bond, m.gnn_params
    )
    assert specs["resources"].num_gates == 3 * n + 3 * e


def test_forward_produces_two_class_logits():
    """Both models return a single (1, 2) logit vector for the classifier."""
    atoms, bonds = preprocess_smiles("CCO")
    assert tuple(IsoQGNN(n_nodes=N_NODES)(atoms, bonds).shape) == (1, 2)
    assert tuple(IsoCGNN(n_nodes=N_NODES)(atoms, bonds).shape) == (1, 2)
