"""
Evaluator of different DTW algorithms. Brings together warping and DTW algorithms
to give a numerical value to the performance of each algorithm.

@author: Jaime Barquero Cebrian
"""

import numpy as np
import warping as wrp
from dtw import dynamic_time_warp


def ev_al(og_contour, anchor=None, width=None, height=None, shift=None, seed=None):
    """
    Evaluates match %, warp %, error % and Max Error of two contours

    Parameters:
        og_contour (array, float): array of frequencies of a contour
        anchor (float): single point that will be shifted, must be between (0,n) exclusive, n = len(og_contour).
        width (int): pass as % of total number of points
        height (float): paper used height of 0, 0.1 and 0.2
        shift (int): how much anchor is shifted by (% of total points)
        seed (int): specific number that allows user to reproduce same randomised values each run

    Returns:
        ev_report (dictionary): match_score, wrp_function, avg_wrp_er, max_wrp_er, avg_wrp and max_wrp
        dist_report (dictionary): anchor, width, height, shift, slopes, seed, redrawn

    Note: ** START and END points are fixed to distort other points **
    """
    n = len(og_contour)
    wrp_contour, key, dist_report = wrp.distort(og_contour, anchor, width, height, shift, seed)

    match_score, wrp_function = dynamic_time_warp(og_contour, wrp_contour)

    if len(wrp_function) != n:
        raise ValueError("Length of warp function differs from length of original contour")

    avg_wrp_er, max_wrp_er = warpAndErrorPercentage(wrp_function, key)
    avg_wrp, max_wrp = warpAndErrorPercentage(wrp_function)

    ev_report = {
        "match_score": match_score,
        "wrp_function": wrp_function,
        "avg_wrp_er": avg_wrp_er,
        "max_wrp_er": max_wrp_er,
        "avg_wrp": avg_wrp,
        "max_wrp": max_wrp,
    }
    return ev_report, dist_report


def warpAndErrorPercentage(wrp_function, key=None):
    """
    Measures the distance between warp and a reference path.

    Parameters:
        wrp_function (array, int): DTW warp path: for each index, the index it matched in the other contour.
        key (array, optional): Warp map returned by wrp.distort. Def: identity.
            - Given: How wrong DTW was compared with the known distortion.
            - None: How much DTW warped the contour overall.

    Returns:
        Avg (float): mean |path - key| as % of n.
        Max_error (float): percentage of maximum error.
    """
    n = len(wrp_function)
    if key is None:
        key = np.arange(n)
    avg = []
    max_error = 0
    for x, y in zip(wrp_function, key):
        cur_diff = abs(x - y)
        if max_error < cur_diff:
            max_error = cur_diff
        avg.append(cur_diff)
    avg = np.mean(avg) * 100 / n
    max_error = max_error * 100 / n

    return avg, max_error
