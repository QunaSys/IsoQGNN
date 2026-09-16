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

"""Graph-construction invariants for preprocess_smiles.

Guards the fixed-width padding (constant register across a batch) and the
bond-type encoding the circuit relies on.
"""
import numpy as np
import pytest

from config import N_MAX
from preprocessing import PADDING_ATOM, preprocess_smiles


def test_hydrogens_are_implicit_heavy_atoms_only():
    atoms, _ = preprocess_smiles("CCO")  # ethanol: 3 heavy atoms
    heavy = [a for a in atoms if a != PADDING_ATOM]
    assert heavy == ["C", "C", "O"]


def test_padding_to_fixed_width():
    atoms, _ = preprocess_smiles("CCO")
    assert len(atoms) == N_MAX
    assert atoms.count(PADDING_ATOM) == N_MAX - 3


def test_bond_type_encoding():
    # single=1, double=2, triple=3, aromatic=4
    assert preprocess_smiles("CCO")[1] == [(0, 1, 1), (1, 2, 1)]
    assert preprocess_smiles("C=O")[1] == [(0, 1, 2)]
    assert preprocess_smiles("C#N")[1] == [(0, 1, 3)]
    assert {t for _, _, t in preprocess_smiles("c1ccccc1")[1]} == {4}


def test_molecule_larger_than_nmax_is_rejected():
    with pytest.raises(Exception):
        preprocess_smiles("C" * (N_MAX + 1))  # 10 heavy atoms > N_MAX


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        preprocess_smiles("not-a-molecule")


def test_train_split_median_labelling_balances_training_set():
    """The labelling rule used by run_experiments.py and run_trivial_baseline.py:
    the threshold is the median of the TRAINING split only (recomputed per
    trial) and is applied unchanged to the held-out molecules, so for distinct
    values the training set is exactly balanced (600/600 of 1,200) while the
    held-out splits are only close to balanced. run_data_efficiency.py applies
    the same idea once per task on its training pool, excluding the fixed test
    set. The rule is replicated here to avoid a network download of QM9."""
    rng = np.random.RandomState(0)
    values = rng.permutation(1500).astype(float)  # 1500 distinct values
    perm = rng.permutation(1500)
    train_idx, held_out_idx = perm[:1200], perm[1200:]

    train_median = np.median(values[train_idx])
    labels = (values > train_median).astype(float)

    assert labels[train_idx].sum() == 600
    assert (labels[train_idx] == 0).sum() == 600
    # Held-out labels use the same threshold; they are close to, not exactly,
    # balanced (150 positives would be exact for the 300 held-out molecules).
    held_out_positives = labels[held_out_idx].sum()
    assert 120 <= held_out_positives <= 180
    assert held_out_positives != 150

    # A threshold taken over all 1,500 values would differ from this one and
    # would not balance the training split.
    whole_median = np.median(values)
    assert whole_median != train_median
    assert (values > whole_median)[train_idx].sum() != 600
