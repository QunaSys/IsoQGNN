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

"""Source-level guards for the protocol invariants the reported results rely on.

These checks read the run scripts as text rather than importing them, since the
scripts download QM9 at import time. They fail loudly if an edit changes the
training budget, the epoch-selection rule, the labelling rule, the matched
capacity, or the dependency pins.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(name):
    return (ROOT / name).read_text()


def test_matched_experiment_trains_to_convergence():
    """The reported metrics are converged 200-epoch values averaged over ten
    trials; the quantum models need the full budget to converge on every seed."""
    src = _src("run_experiments.py")
    epochs = re.findall(r"^N_EPOCHS\s*=\s*(\d+)\b", src, re.M)
    trials = re.findall(r"^N_TRIALS\s*=\s*(\d+)\b", src, re.M)
    assert epochs == ["200"], f"expected a single N_EPOCHS = 200, found {epochs}"
    assert trials == ["10"], f"expected a single N_TRIALS = 10, found {trials}"


def test_data_efficiency_selects_on_validation_not_test():
    """For each training-set size the reported epoch is chosen on validation
    AUC, applied identically to both architectures, so the test set is scored
    once and never searched over."""
    src = _src("run_data_efficiency.py")
    # A validation loader, and the reported epoch taken from the validation
    # history: exactly one assignment of best_ep, and it reads val_hist.
    assert "val_loader" in src
    best_ep = re.findall(r"^\s*best_ep\s*=\s*(.+)$", src, re.M)
    assert len(best_ep) == 1, f"expected one best_ep assignment, found {best_ep}"
    assert "val_hist" in best_ep[0], f"epoch not selected on validation: {best_ep[0]}"
    # Nothing may search a test-set history, by argmax/max or a running best.
    assert not re.search(r"(argmax|max)\(\s*test_", src), "selects on test history"
    assert not re.search(r">\s*best_\w*auc", src), "running best-test-AUC tracker"


def test_labels_use_train_split_median_only():
    """Labels come from training molecules only: each run script computes its
    median-split threshold on its own training molecules, never on rows of the
    held-out test set. run_experiments.py and run_trivial_baseline.py use the
    per-trial training split; run_data_efficiency.py uses its training pool,
    which excludes the fixed 500-molecule test set."""
    sanctioned = {
        "run_experiments.py": r"np\.median\(\s*valid_vals\s*\[\s*train_idx\s*\]\s*\)",
        "run_trivial_baseline.py": r"np\.median\(\s*vals\s*\[\s*tr\s*\]\s*\)",
        "run_data_efficiency.py": r"np\.median\(\s*valid_vals\s*\[\s*train_pool_idx\s*\]\s*\)",
    }
    for name, pattern in sanctioned.items():
        src = _src(name)
        assert re.search(pattern, src), f"{name}: train-only median call missing"
        # The sanctioned call must be the ONLY median taken in the script.
        assert len(re.findall(r"\bmedian\(", src)) == 1, f"{name}: extra median() call"
        # No alternative estimator may compute a threshold over all molecules.
        for banned in ("statistics.median", ".median()", "percentile(", "quantile("):
            assert banned not in src, f"{name}: uses {banned}"


def test_config_is_single_layer_cnof_nmax9():
    """The 64-parameter matched capacity depends on these config invariants."""
    src = _src("config.py")
    assert re.search(r"^N_LAYERS\s*=\s*1\b", src, re.M)
    assert re.search(r"^N_MAX\s*=\s*9\b", src, re.M)
    assert re.search(r'ATOM_TYPES\s*=\s*\[\s*"C",\s*"N",\s*"O",\s*"F"\s*\]', src)


def test_requirements_are_exactly_pinned():
    """Reproducibility of the seed-sensitive quantum results relies on exact
    dependency pins, not floors."""
    lines = [
        ln.strip()
        for ln in _src("requirements.txt").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert lines, "requirements.txt has no dependency lines"
    for ln in lines:
        assert "==" in ln and ">=" not in ln, f"not exactly pinned: {ln}"
