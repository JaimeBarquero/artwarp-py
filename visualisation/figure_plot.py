"""
Generate a figure with 3 graphs. Highilighting DTW incorrect compensation for y-axis distortion (Gaussian Hill)

@author: Jaime Barquero Cebrian
"""
import sys
from pathlib import Path

import matplotlib
# Non-Interactive plot. Agg is to produce PNG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import warping as wrp
from dtw import dynamic_time_warp, compute_similarity_matrix

def main():
    """ uv run --with matplotlib python y-distort-graph.py out.png """
    output_path = Path('/Users/jaimebarquero/Downloads')/sys.argv[1]

    # Contour and chosen warping
    og = wrp.csvLoader(Path(__file__)/ "data" / "F109-2001-SW-IND63104_ROCCA.csv")
    bumped = wrp.yDistortion(og, 100, 100, 8, 0.15)

    # Apply DTW
    match_y, warp_y = dynamic_time_warp(og, bumped)

    build_graph(output_path, 7000, 0.15, 100, match_y, warp_y, og, bumped, True)

    # Figure on baseline (DTW applied on non-distorted contour), shows baseline error
        # match_no_dist, warp_no_dist = dynamic_time_warp(og, og)
        # build_graph(output_path, 7000, 0, 100, match_no_dist, warp_no_dist, og, og, True)

def build_graph(out, offset, height, anchor, match, warp, og_contour, distorted_contour, show_ties=False):
    """
    Creates a figure of 3 graphs A,B,C.
    A: Shows the original contour and the distorted contour in the same graph.
    B: Shifts the distorted contour down (purely for visualisation), and shows the point matching for DTW across both contours.
    C: Highlights the true similarity between both contours. A good TW algorithm would give the same % similarity as the true path.

    Parameters:
        out (Path): output path of figure.
        offset (int (Hz)): shifts the distorted contour down in panel B to visualise DTW more clearly.
        height (float): Gaussian hill height.
        anchor (int): index chosen as anchor when distorting.
        match (%, int): % similarity given by time warping.
        warp (array, int): array given by time warping.
    """
    path = check_output_path(out)
    n = len(og_contour)
    idx = np.arange(n)

    # Computes similarity matrix. Each element is the % similarity between points [a,b].
    S = compute_similarity_matrix(og_contour, distorted_contour)
    # The diagonal of the similarity matrix. 
    # Each original point compared with the distorted points at the same time.
    true_sim = S[idx, idx]
    # Each original point compared with the point the TW algorithm thought was best.
    dtw_sim = S[idx, warp]

    blue, orange, grey, red = "#1f77b4", "#ff7f0e", "#b0b0b0", "#d62728"
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(21, 6))

    # Plot A
    a.plot(idx, og_contour, color=blue, label="original")
    a.plot(idx, distorted_contour, color=orange, label=f"y-bump {height} at index {anchor}")
    a.set_title("A. Distortion on the y-axis", fontsize=14)
    a.set_xlabel("point index", fontsize=12)
    a.set_ylabel("Hz", fontsize=12)
    a.legend(loc="lower center")

    # Plot B
    low = distorted_contour - offset
    # Grey/Red lines
    for i in range(0, n, 2):
        j = warp[i]
        tie = og_contour[j] == og_contour[i]
        warped = j != i and (show_ties or not tie)
        b.plot([i, j], [og_contour[i], low[j]], color=red if warped else grey,
            lw=0.9 if warped else 0.6, alpha=0.9 if warped else 0.6, zorder=1)

    b.plot(idx, og_contour, color=blue, zorder=2, label="original")
    b.plot(idx, low, color=orange, zorder=2, label="y-bump")
    b.plot([], [], color=grey, label="DTW match, same time or tie (same original value)")
    b.plot([], [], color=red, label="DTW match, warped to a different value")
    b.set_title("B. DTW warps X-axis to compensate", fontsize=14)
    b.set_xlabel("point index", fontsize=12)
    b.set_ylabel("Hz", fontsize=12)
    b.set_yticklabels([])
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=10, frameon=False)

    # Plot C
    c.plot(idx, true_sim, color=blue, label=f"true path, mean {true_sim.mean():.2f}%")
    c.plot(idx, dtw_sim, color=orange, label=f"DTW path, mean {dtw_sim.mean():.2f}%")
    c.set_title("C. Wrong path scores higher", fontsize=14)
    c.set_xlabel("original point", fontsize=12)
    c.set_ylabel("per-point similarity %", fontsize=12)
    c.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path)

def check_output_path(path):
    counter = 1
    if path.exists():
        while True:
            new_name = f"{path.stem}_{counter}{path.suffix}"
            if not (path.parent/new_name).exists():
                return path.parent/new_name
            counter += 1
    return path

if __name__ == "__main__":
    main()



