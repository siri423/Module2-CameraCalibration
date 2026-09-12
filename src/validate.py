"""
================================================================================
 validate.py  --  STEP 3: Validation experiment + error statistics
================================================================================
 CSc 8830 Computer Vision, Module 2.

 WHAT IT DOES
   Reads a CSV log of >= 20 measurements. For each row it computes the object
   size estimated by measure.py and compares it against the TRUE (tape-measured)
   size, then reports error statistics and saves plots.

 INPUT CSV (measurements_log.csv) -- one row per measurement:
   image, object, dimension, true_size_mm, distance_mm, x1, y1, x2, y2
     image        : filename in data/validation_images (for your reference)
     object       : e.g. "book", "box"
     dimension    : e.g. "width" / "height"
     true_size_mm : real size measured with a tape/ruler  (GROUND TRUTH)
     distance_mm  : camera->object distance, tape-measured, > 2000 mm
     x1,y1,x2,y2  : pixel endpoints of that dimension in the image
                    (the web app fills these in when you click the two ends)

 HOW TO RUN
   python validate.py --calib ../outputs/calibration.npz \
                      --log ../measurements_log.csv \
                      --out ../outputs/results

 OUTPUT
   results/validation_results.csv  (per-row estimate, error, % error)
   results/error_stats.txt         (summary statistics)
   results/error_plots.png         (estimated-vs-true + error histogram)
================================================================================
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_calibration
from measure import measure_length


def run(calib_path, log_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c = load_calibration(calib_path)
    K, dist = c["camera_matrix"], c["dist_coeffs"]

    rows = []
    with open(log_path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("true_size_mm"):        # skip blank/comment lines
                continue
            true = float(r["true_size_mm"])
            Z = float(r["distance_mm"])
            p1 = (float(r["x1"]), float(r["y1"]))
            p2 = (float(r["x2"]), float(r["y2"]))
            est, _, _ = measure_length(p1, p2, Z, K, dist)
            err = est - true
            rows.append({
                "image": r.get("image", ""),
                "object": r.get("object", ""),
                "dimension": r.get("dimension", ""),
                "true_mm": true,
                "distance_mm": Z,
                "est_mm": est,
                "error_mm": err,
                "abs_error_mm": abs(err),
                "pct_error": 100.0 * abs(err) / true if true else float("nan"),
            })

    if not rows:
        raise SystemExit("No valid rows in the log. Fill in measurements_log.csv.")

    # ---- write per-row results ----
    res_csv = os.path.join(out_dir, "validation_results.csv")
    with open(res_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- statistics ----
    errs = np.array([r["error_mm"] for r in rows])
    abse = np.array([r["abs_error_mm"] for r in rows])
    pcte = np.array([r["pct_error"] for r in rows])
    stats = [
        f"Number of measurements     : {len(rows)}",
        f"Mean error (bias)          : {errs.mean():.2f} mm",
        f"Std of error               : {errs.std(ddof=1):.2f} mm",
        f"Mean absolute error (MAE)  : {abse.mean():.2f} mm",
        f"RMSE                       : {np.sqrt((errs**2).mean()):.2f} mm",
        f"Max absolute error         : {abse.max():.2f} mm",
        f"Mean abs percentage error  : {pcte.mean():.2f} %",
        f"Median abs percentage error: {np.median(pcte):.2f} %",
    ]
    stats_txt = "\n".join(stats)
    with open(os.path.join(out_dir, "error_stats.txt"), "w") as f:
        f.write("VALIDATION ERROR STATISTICS\n===========================\n")
        f.write(stats_txt + "\n")
    print("\n" + stats_txt)

    # ---- plots ----
    true_vals = np.array([r["true_mm"] for r in rows])
    est_vals = np.array([r["est_mm"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    lim = [0, max(true_vals.max(), est_vals.max()) * 1.1]
    ax[0].plot(lim, lim, "k--", label="perfect")
    ax[0].scatter(true_vals, est_vals, c="tab:blue")
    ax[0].set_xlabel("True size (mm)"); ax[0].set_ylabel("Estimated size (mm)")
    ax[0].set_title("Estimated vs. true"); ax[0].legend(); ax[0].set_aspect("equal")
    ax[1].hist(errs, bins=12, color="tab:orange", edgecolor="black")
    ax[1].axvline(0, color="k", ls="--")
    ax[1].set_xlabel("Error (mm)"); ax[1].set_ylabel("Count")
    ax[1].set_title("Error distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_plots.png"), dpi=130)
    print(f"\nSaved results -> {out_dir}")
    return rows, stats_txt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 3: validation + error stats")
    p.add_argument("--calib", default="../outputs/calibration.npz")
    p.add_argument("--log", default="../measurements_log.csv")
    p.add_argument("--out", default="../outputs/results")
    a = p.parse_args()
    run(a.calib, a.log, a.out)
