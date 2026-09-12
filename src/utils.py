"""
utils.py  --  Shared helpers for loading/saving calibration results.

These functions are imported by calibrate.py, measure.py, validate.py and app.py.
Nothing here is run directly.
"""
import json
import os
import numpy as np


def save_calibration(path, camera_matrix, dist_coeffs, image_size,
                     rms, square_size_mm, pattern_size):
    """Save calibration to a .npz (binary) and a .json (human readable)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=np.array(image_size),
             rms=rms,
             square_size_mm=square_size_mm,
             pattern_size=np.array(pattern_size))
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.ravel().tolist(),
            "image_size": list(image_size),
            "reprojection_rms_px": float(rms),
            "square_size_mm": float(square_size_mm),
            "pattern_size_internal_corners": list(pattern_size),
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        }, f, indent=2)
    return json_path


def load_calibration(path):
    """Load a .npz calibration file into a dict."""
    d = np.load(path, allow_pickle=True)
    return {
        "camera_matrix": d["camera_matrix"],
        "dist_coeffs": d["dist_coeffs"],
        "image_size": tuple(int(x) for x in d["image_size"]),
        "rms": float(d["rms"]),
        "square_size_mm": float(d["square_size_mm"]),
        "pattern_size": tuple(int(x) for x in d["pattern_size"]),
    }
