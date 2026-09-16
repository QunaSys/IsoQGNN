# Conference talk

Slides for the QCE26 presentation of *Implementations of Quantum and Classical
Topology-Aligned Architectures for Molecular Property Prediction*, given at IEEE
Quantum Week 2026 in Toronto on 15 September 2026, in the *Quantum ML for
Chemistry and Molecules* session.

- **[IsoQGNN-QCE26-Toronto.pdf](IsoQGNN-QCE26-Toronto.pdf)** — the deck as presented.
- **[figures/](figures/)** — every figure in the deck, named by the slide it appears on.
- **[make_slide_figures.py](make_slide_figures.py)** — the single generator for those figures.

## Regenerating the figures

Every figure is built by one script, and every number on them is read at build
time from the result files committed at the root of this repository
(`experiment_summary_metrics.csv`, `experiment_full_curves.csv`,
`data_efficiency_combined.csv`, `trivial_baseline_metrics.csv`). So the slides
can be checked against the published data rather than taken on trust:

```bash
python talk/make_slide_figures.py
```

The dataset figure additionally needs the QM9 source table. It is downloaded
once, from the same URL `run_experiments.py` uses, and cached in
`figures/.data/`, which is not tracked here.

Figures are styled for projection rather than for print, so they use a larger
sans-serif face than the figures in the paper. The two sets are drawn from the
same result files and agree numerically.
