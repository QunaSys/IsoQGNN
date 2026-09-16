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

"""Torch Dataset wrapping (SMILES, label) pairs with lazy preprocessing."""
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    """Pairs SMILES strings with labels and preprocesses on access.

    Each item is a dict ``{"atoms", "bonds", "label"}`` produced by applying
    ``preprocess_fn`` to the SMILES on retrieval (not up front), so the graph
    representation is built lazily per sample.
    """

    def __init__(self, smiles_list, labels, preprocess_fn):
        """
        Args:
            smiles_list (List[str]): SMILES strings
            labels (List[int]): binary or multiclass labels
            preprocess_fn (callable): preprocess_smiles(smiles) -> (atoms, bonds)
        """
        assert len(smiles_list) == len(
            labels
        ), f"Mismatched data lengths: {len(smiles_list)} SMILES vs {len(labels)} Labels"

        self.smiles = smiles_list
        self.labels = labels
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        atoms, bonds = self.preprocess_fn(smiles)

        return {
            "atoms": atoms,
            "bonds": bonds,
            "label": label,
        }
