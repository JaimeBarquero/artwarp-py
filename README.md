# DTW Evaluation
**Evaluator for comparing different DTW algorithms before integrating them to artwarp-py**

## Overview

[artwarp-py](https://github.com/dolphin-acoustics-vip/artwarp-py) uses Dynamic Time Warping (DTW) to compare dolphin whistle contours, but its current DTW struggles when contours differ in frequency (y-axis) rather than timing (x-axis). This project measures how well each DTW variant recovers a **known** warp: a real contour is distorted on the x-axis (time shift) and y-axis (Gaussian frequency bump) by a known amount, the DTW algorithm is run on original vs distorted, and the warp it finds is compared with the true one. Each run reports a match score, how much the algorithm warped, and how far its warp was from the true warp.

Current DTW algotithms being evaluated:
    -ARTwarp dtw: Current base algorithm (artwarp-py), the one that will be replaced. Lacks robustness against yDistortions.

Algorthims to implement:
- Derivative Dynamic Time Warping (DDTW): Uses only derivatives, extremely useful for yDistortions. Start and end points cannot be evaluated (uses right and left neighbour for slope). Ignores absolute Hz values.

- Value-Derivative Dynamic Time Warping (VDDTW): Mixture of current value based DTW and DDTW, in theory could be really useful. Equation in research docs uses product which means it still disregards absolute Hz values.

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Distortion Parameters](#distortion-parameters)
- [Metrics](#metrics)
- [Results](#results)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [References](#references)
- [Author](#author)

---

## Installation

**Requirements:** Python 3.8+, [uv](https://docs.astral.sh/uv/). Dependencies: NumPy ≥1.21, Numba ≥0.54.

```bash
# clone and set up environment (creates .venv from uv.lock)
> git clone <repo-url>
> cd dtw-evaluation
> uv sync

# without uv
> python -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
```

---

## Quick Start

Run the default evaluation (four distortion cases on the sample contour):

```bash
> uv run main.py
```

```
Run                    Match    Warp %   Avg Error %   Max Error %
No Distortion         100.00    0.18        0.18          3.30
Only X Distorted       99.69    4.09        0.26          3.65
Only Y Distorted       99.13    1.21        1.21          7.08
Both X,Y Distorted     99.00    3.57        1.35          7.97
```

Evaluate a single distortion from Python:

```python
from pathlib import Path
import warping as wrp
import evaluate

contour = wrp.csvLoader(Path("data") / "F109-2001-SW-IND63104_ROCCA.csv")

# any parameter left as None is randomised from seed
ev_report, dist_report = evaluate.ev_al(contour, anchor=100, width=8, height=0.15, shift=8, seed=6)

print(ev_report["match_score"], ev_report["avg_wrp_er"])  # 98.99..., 1.35...
print(dist_report)  # anchor, width, height, shift, slopes, seed, redrawn
```

---

## How It Works

1. Contour is warped on both axes (TODO: possibility of warping on single axis). (warping.py).
    For distortion end points are fixed to ensure the underlying whistle remains the same. Slope must be between [1/3, 3] for ARTwarp DTW to function.
        (TODO: Once other algorithms are implemented, they shouldnt really be restricted to this slope, unless comparing agains DTW)
    - xDistortion: Distorts the contour x-axis by shifting anchor point and proportionaly streching/squashing all other points.
    - yDistortion: Distorts the contour y-axis by creating a hill in the contour, using the standard Gaussian hill formula.
    - distortion: Combines both distortions, gives randomised values for distortion if not specified. Has `seed` parameter to reproduce randomised values.
2. Distorted contour and original are evaluated (`evaluate.py`) and display
match, warp %, error % and max error between original contour and the distorted one after applying DTW algorithm.

---

## Distortion Parameters

`n` = number of points in the contour. Parameters left as `None` in `distort` / `ev_al` are randomised from `seed`.

| Parameter | Meaning | Units | Random range | Constraint |
|-----------|---------|-------|--------------|------------|
| `anchor` | Point that is shifted on the x-axis; also centre of the y-bump | index | integer in [0.1n, 0.9(n−1)) | 0 < anchor < n−1 |
| `shift` | How far the anchor moves | % of n | uniform(−15, 15) | 0 < anchor+shift < n−1; resulting slopes in [1/3, 3] |
| `width` | Standard deviation of the Gaussian bump | % of n | uniform(2, 15) | > 0 |
| `height` | Peak bump size, relative to frequency (0.15 = +15% at the peak) | fraction | uniform(0, 0.2) | ≥ 0 |
| `seed` | Seed for the random generator, so randomised runs can be reproduced | int | random entropy | — |

If a **random** shift gives an invalid slope, a new shift is drawn (max 5 attempts, `redrawn = True` in the report). If a **user-chosen** shift is invalid, a `ValueError` is raised instead.

---

## Metrics

All warp metrics are percentages of `n`, so contours of different lengths can be compared.

| Key | Definition | Good value |
|-----|------------|------------|
| `match_score` | ARTwarp similarity: average point-wise similarity along the DTW path (0–100) | High |
| `avg_wrp` / `max_wrp` | Mean / max of \|warp_function[i] − i\|: how much DTW warped the contour overall | Close to the true warp |
| `avg_wrp_er` / `max_wrp_er` | Mean / max of \|warp_function[i] − key[i]\|: how far the DTW warp is from the known (true) warp | Low |

**Baseline:** with no distortion the error is 0.18%, not 0. TODO: FIX base error.

---

## Results

### ARTwarp DTW

Example run:

| Run | anchor | width | height | shift | seed | Match | Warp % | Avg Error % | Max Error % |
|-----|--------|-------|--------|-------|------|-------|--------|-------------|-------------|
| No distortion | random | random | 0 | 0 | 6 | 100.00 | 0.18 | 0.18 | 3.30 |
| Only X | 100 | 8 | 0 | 8 | — | 99.69 | 4.09 | 0.26 | 3.65 |
| Only Y | 100 | 8 | 0.15 | 0 | 6 | 99.13 | 1.21 | 1.21 | 7.08 |
| Both X, Y | 100 | 8 | 0.15 | 8 | 6 | 99.00 | 3.57 | 1.35 | 7.97 |

Y-distortion causes most of the error: with only a frequency bump (no true warp), DTW still warps 1.21% on average and up to 7.08%, in other words, it explains a y-change by warping the x-axis. X-only distortion is recovered well (0.26% avg error). Match score stays above 99% in every case, so it does not reveal the misalignment by itself.

---

## Project Structure

```
dtw-evaluation/
├── main.py            # Entry point: runs the distortion cases and prints results table
├── evaluate.py        # ev_al: distort → DTW → metrics; warpAndErrorPercentage
├── warping.py         # csvLoader, xDistortion, yDistortion, distort, redraw_planGraph
├── dtw.py             # ARTwarp DTW (from artwarp-py, Itakura constraint, optional Numba)
├── data/              # Sample whistle contours (.csv)
├── tests/
│   └── test_warping.py
├── pyproject.toml     # Project metadata, dependencies, tool config
├── uv.lock            # Exact pinned versions
└── requirements.txt   # pip fallback
```

---

## Testing

```bash
> uv run pytest
```

Covers `warping.py`: CSV loading, `xDistortion` output and invalid anchor/shift errors, `yDistortion` output, and `distort`.

---

## References

1. **artwarp-py:** Gronda Garrigues, P. et al. (2026). artwarp-py (v1.0.2). Zenodo. https://doi.org/10.5281/zenodo.23551892
2. **Deecke, V. B. & Janik, V. M. (2006).** Automated categorization of bioacoustic signals: Avoiding perceptual pitfalls. *J. Acoust. Soc. Am.* **119**, 645–653. https://doi.org/10.1121/1.2139067
3. **Keogh, E. J. & Pazzani, M. J. (2001).** Derivative Dynamic Time Warping. *Proc. SIAM International Conference on Data Mining.*
4. **Kulbacki, M. & Bąk, A. (2002).** Unsupervised learning motion models using Dynamic Time Warping. *Intelligent Information Systems 2002*, 217–226.
5. **Górecki, T. & Łuczak, M. (2014).** Non-isometric transforms in time series classification using DTW. *Knowledge-Based Systems* **61**, 98–108.

---

## Author

- **Evaluator:** Jaime Barquero Cebrian (2026)
- **`dtw.py`:** Pedro Gronda Garrigues, from [artwarp-py](https://github.com/dolphin-acoustics-vip/artwarp-py) (LGPL-3.0)
