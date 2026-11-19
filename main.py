"""
Main endpoint to evaluate a distortion and create table with evaluator results.

@author: Jaime Barquero Cebrian
"""

import warping as wrp
from pathlib import Path
import evaluate

if __name__ == "__main__":

    def printReport(*results):
        print(f"{'Run':20}{'Match':>8}{'Warp %':>10}{'Avg Error %':>14}{'Max Error %':>14}")
        labels, distortions = results
        for label, distortion in zip(labels, distortions):
            ev_report = distortion[0]

            print(
                f"{label:20}{ev_report['match_score']:8.2f}{ev_report['avg_wrp']:8.2f}{ev_report['avg_wrp_er']:12.2f}{ev_report['max_wrp_er']:14.2f}"
            )

    def possDistortions(contour):
        no_dist = evaluate.ev_al(contour, height=0, shift=0, seed=6)
        only_x_dist = evaluate.ev_al(contour, anchor=100, width=8, height=0, shift=8)
        only_y_dist = evaluate.ev_al(contour, anchor=100, width=8, height=0.15, shift=0, seed=6)
        both_dist = evaluate.ev_al(contour, anchor=100, width=8, height=0.15, shift=8, seed=6)

        labels = ["No Distortion", "Only X Distorted", "Only Y Distorted", "Both X,Y Distorted"]
        distortions = [no_dist, only_x_dist, only_y_dist, both_dist]
        return labels, distortions

    contour = wrp.csvLoader(Path(__file__).parent / "data" / "F109-2001-SW-IND63104_ROCCA.csv")
    labels, distortions = possDistortions(contour)
    printReport(labels, distortions)

    # Baseline error of 0.18%
