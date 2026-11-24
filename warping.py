"""
Distorts a given contour on the x-axis and on the y-axis. If distortion is invalid for ARTwarp DTW,
it is redrawn a maximum of 5 times before failing.

@author: Jaime Barquero Cebrian
"""

import numpy as np


def csvLoader(path):
    freq = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
    return freq


def xDistortion(freq, anchor, shift=None):
    """
    Distorts the contour x-axis by shifting anchor point and proportionaly streching/squashing all other points

    Parameters:
        freq (array, float): array of frequencies of a contour
        anchor (float): single point that will be shifted, must be between 0 and len(freq) (not including).
        shift (int): how much anchor is shifted by (% of total points)
        Note: ** START and END points are fixed to distort other points **

    Returns:
        planGraph (array, float): depends solely on anchor, len(freq) and shift.
            Graph with original index vs new index, with no shift it is a diagonal through 0,0.
            After shifting anchor the new index shows what old index will be used to fetch a frequency.
            It is an array of the original positions after shift.
        distorted (array, float): Array of new frequencies that have been streched or compressed.
            len(freq) is maintained.
        key (array, float): same format as DTW's warp_function.
            Used to find where each old frequency is with respect to new frequencies.
        slopes (array, float):
            slopes of lines from (0,0) -> (anchor+shift, anchor) and (anchor+shift, anchor) -> (n-1, n-1) on planGraph

    Raises:
        ValueError: if anchor is outside (0, n-1)
        ValueError: if anchor+shift is outside (0, n-1)
    
    Note: ** Look at documentation for full detailed explanation of xDistortion **.
    Note: None is only used for direct testing, it will always have an assigned value.
    """
    n = len(freq)
    shift = n * shift / 100

    if anchor <= 0 or anchor >= n - 1:
        raise ValueError(f"{anchor} outside (0, {n-1})")
    if 0 >= anchor + shift or anchor + shift >= n - 1:
        raise ValueError(f"{anchor+shift} outside (0, {n-1})")

    point_x = [0, anchor + shift, n - 1]  # Distorted
    point_y = [0, anchor, n - 1]  # Original

    slopes = []
    prev_x, prev_y = point_x[0], point_y[0]
    for x, y in zip(point_x, point_y):
        if x == 0 and y == 0:
            continue
        slopes.append((y - prev_y) / (x - prev_x))
        prev_x, prev_y = x, y

    for slope in slopes:
        if slope < 1 / 3 or slope > 3:
            raise ValueError(
                f"slope {slope} outside [1/3, 3]."
                "Derived from anchor = {anchor}, shift = {shift:.2f} pts, using shift ({shift*100/n:.1f}%). "
                "Please increase anchor and/or decrease shift"
            )

    planGraph = np.interp(np.arange(0, n), point_x, point_y)
    key = np.interp(np.arange(0, n), point_y, point_x)  # Swapped axes: x = original, y = distorted
    distorted = np.interp(planGraph, np.arange(0, n), freq)

    for i in range(0, len(slopes)):
        slopes[i] = float(slopes[i])

    return planGraph, distorted, key, slopes


def yDistortion(freq, anchor, centre=None, width=None, height=None):
    """
    Distorts the contour y-axis using standard Gaussian hill formula.

    Parameters:
        freq (array, float): array of frequencies of a contour
        anchor (float): single point that will be bumped, must be between 0 and len(freq) (not including).
        centre (float): specific point
        width (int): pass as % of total number of points
        height (float): paper used height of 0, 0.1 and 0.2
        Note: None is only used for direct testing, it will always have an assigned value

    Returns:
        bump (array, float): array of all the contour freq values with the applied distortion

    Raises:
        ValueError: if width is <= 0
    """
    n = len(freq)
    if centre is None:
        centre = anchor
    width = 0.06 * n if width is None else (n * width / 100)
    if width <= 0:
        raise ValueError(f"Width: {width} <= 0 is invalid")
    if height is None:
        height = 0.15  #  so smth in between paper chosen vals

    weight = np.exp(-0.5 * ((np.arange(n) - centre) / width) ** 2)
    bump = freq * (1 + height * weight)

    return bump


def distort(freq, anchor=None, width=None, height=None, shift=None, seed=None):
    """
    Combines xDistort and yDistortion to fully distort a contour.
    Randomises parameters with a specific seed (for reproduceability) if they are not specified.

    Parameters:
        freq (array, float): array of frequencies of a contour
        anchor (float): single point that will be shifted, must be between (0,n) exclusive,
            n = len(freq).
        width (int): pass as % of total number of points
        height (float): paper used height of 0, 0.1 and 0.2
        shift (int): how much anchor is shifted by (% of total points)
        seed (int): specific number that allows user to reproduce same randomised values each run
        Note: ** START and END points are fixed to distort other points **

    Returns:
        distorted (array, float): array of distorted frequencies
        key (array, float): same format as DTW's warp_function.
            Used to find where each old frequency is with respect to new frequencies.
        report (dictionary): anchor, width, height, shift, slopes, seed, redrawn
    """
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rnd = np.random.default_rng(seed)
    n = len(freq)
    user_shift = shift

    # Randomise prameters if some/none are not specified. High boundary is not-inclusive on .integers
    if width is None:
        width = rnd.uniform(2, 15)
    if height is None:
        height = rnd.uniform(0, 0.2)
    # Took 10% off from start and 10% off from end to avoid fixed end points before and after bump
    if anchor is None:
        anchor = rnd.integers(int(n - n * 0.9), int((n - 1) * 0.9))
    if shift is None:
        shift = rnd.uniform(-15, 15)

    # Distort original centered at the anchor bump travels with anchor
    ydistort = yDistortion(freq, anchor, anchor, width, height)
    xdistorted, redrawn, shift = redraw_planGraph(ydistort, anchor, shift, rnd, user_shift)
    distorted, key, slopes = xdistorted[1], xdistorted[2], xdistorted[3]

    report = {
        "anchor": int(anchor),
        "width": float(width),
        "height": float(height),
        "shift": shift,
        "slopes": slopes,
        "seed": seed,
        "redrawn": redrawn,
    }
    return distorted, key, report


def redraw_planGraph(ydistort, anchor, shift, rnd, user_shift=None, attempts=None):
    """
    Method helper in case resulting planGraph from xDistortion has a slope outside of the interval (1/3, 3).
    Retries xDistortion for a maximum of 5 attemps, with new randomised shift value for each attempt.
    Raises ValueError if all attempts are consumed or if user shift value resulted in erroneous slope.

    Parameters:
        ydistort (array, float): ydistortion of the contour.
        anchor (float): single point that will be shifted, must be between 0 and len(freq) (not including).
        shift (int): how much anchor is shifted by (% of total points) (def: 6%)
        rnd (Generator): generator for random values
        user_shift: user chosen shift (None for randomised shift)
        attempts (int): used for testing to reproduce randomised parameters. (def: 5)

    Returns:
        xDistortion: tuple(planGraph, distorted, key, slopes)
        redrawn (boolean): if planGraph had to be redrawn because of invalid resulting slope
        shift: final shift used

    Raises:
        ValueError: Too many attempts to redraw graph
        ValueError: User chose wrong shift as input

    """
    if attempts is None:
        attempts = 5
    if attempts == 0:
        raise ValueError("Too many attempts to redraw graph. Please run again")
    try:
        redrawn = False if attempts == 5 else True
        return (xDistortion(ydistort, anchor, shift), redrawn, shift)
    except ValueError:
        # If the user chose a wrong shift dont redraw
        if user_shift is not None and attempts == 5:
            raise
        if anchor == 0:
            raise
        # Random values returned improper slope redraw
        shift = rnd.uniform(-15, 15)
        return redraw_planGraph(ydistort, anchor, shift, rnd, attempts=attempts - 1)
