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

# config.py
"""
Configuration Constants for Quantum GNN (QGNN)
==============================================

This module defines global hyperparameters and structural constraints used across
data preprocessing, model architecture, and training pipelines. Centralizing these
constants ensures consistency between the dataset generation (preprocessing.py) and
the quantum circuit definition (qgnn.py).

Constants:
    ATOM_TYPES (List[str]):
        The vocabulary of allowed atomic symbols.
        - Used for one-hot encoding node features.
        - Any molecule containing an atom NOT in this list will be filtered out.
        - Current set includes QM9-standard atoms (C, N, O, F).

    N_MAX (int):
        The maximum allowable number of heavy atoms (non-hydrogen) per molecule.
        - Molecules larger than this are discarded during preprocessing.
        - Determines the total number of qubits: n_qubits = N_MAX * QUBITS_PER_NODE.
        - Current value (9) aligns with the standard QM9 definition.

    QUBITS_PER_NODE (int):
        The number of qubits allocated to represent a single atom's features.
        - Higher values increase expressivity but drastically increase simulation cost.
        - Cost scales as 2^(N_MAX * QUBITS_PER_NODE).
        - Set to 1 for efficiency (total qubits = 9).

    N_LAYERS (int):
        The depth of the quantum circuit (number of repeated variational layers).
        - More layers allow for more complex entanglement but are harder to train
          (barren plateaus) and slower to simulate.

    N_CLASSES (int):
        The number of output neurons in the final classical classifier.
        - Set to 2 for binary classification (e.g., Low Gap vs. High Gap) when
          using CrossEntropyLoss.
        - Set to 1 if using BCEWithLogitsLoss.

    LEARNING_RATE (float):
        Adam learning rate shared by both neural training scripts.
        - Each of those scripts sets its own epoch budget alongside this.
"""


ATOM_TYPES = ["C", "N", "O", "F"]
N_MAX = 9
QUBITS_PER_NODE = 1
N_LAYERS = 1
N_CLASSES = 2

LEARNING_RATE = 0.01
