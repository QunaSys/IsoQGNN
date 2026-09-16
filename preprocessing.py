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
SMILES -> fixed-width molecular graph.

Converts a SMILES string into the (atoms, bonds) representation both
architectures consume. Hydrogens are implicit (heavy atoms only); the atom
list is padded to a constant width so every molecule occupies the same
register; bond types are encoded as integers (single=1, double=2, triple=3,
aromatic=4).
"""
from rdkit import Chem

from config import N_MAX

PADDING_ATOM = "PAD"


def preprocess_smiles(smiles, n_max=N_MAX):
    """Parse ``smiles`` into (atoms, bonds).

    Returns:
        atoms: list of heavy-atom symbols, length ``n_max``, right-padded with
            PADDING_ATOM.
        bonds: list of ``(i, j, bond_type)`` triples with integer bond types;
            only bonds whose endpoints fall within the register are kept.

    Raises:
        ValueError: if ``smiles`` cannot be parsed by RDKit.
        Exception: if the molecule has more than ``n_max`` heavy atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    pad_len = n_max - len(atoms)

    # 1. Error if too large.
    if pad_len < 0:
        raise Exception(f"Invalid N_MAX {N_MAX} length: {smiles}")
    # 2. Pad if too small
    elif pad_len > 0:
        atoms += [PADDING_ATOM] * pad_len

    # Edge features
    bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()

        # Convert to int mapping
        if bt == Chem.rdchem.BondType.SINGLE:
            bond_type = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            bond_type = 2
        elif bt == Chem.rdchem.BondType.TRIPLE:
            bond_type = 3
        elif bt == Chem.rdchem.BondType.AROMATIC:
            bond_type = 4
        else:
            bond_type = 1  # fallback to single

        # Ensure valid indices within truncated/padded range
        if i < n_max and j < n_max:
            bonds.append((i, j, bond_type))

    return atoms, bonds
