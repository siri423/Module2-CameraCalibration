"""
================================================================================
 measure.py  --  STEP 2: Real-world 2D size of an object from one photo
================================================================================
 CSc 8830 Computer Vision, Module 2.

 THE IDEA (perspective projection / pinhole model)
   A pinhole camera projects a 3D point onto the image as:
         u = fx * X/Z + cx        v = fy * Y/Z + cy
   Rearranged, a real length that lies in a plane facing the camera at depth Z
   maps to pixels as:   (pixels) = (focal_length_px) * (real_length) / Z
   So going the other way:
         real_length = pixel_length * Z / focal_length_px

   For a segment measured between two image points, we handle the x and y parts
   with fx and fy separately and combine:
         real_dx = dx_px * Z / fx
         real_dy = dy_px * Z / fy
         real_length = sqrt(real_dx^2 + real_dy^2)

 ASSUMPTIONS (state these in your report)
   * The measured face of the object is roughly parallel to the image plane
     (facing the camera flat), so its depth is ~constant = Z.
   * Z is the distance from the camera to the object along the viewing axis,
     measured with a tape measure.
   * Points are undistorted first using the calibrated distortion coefficients.

 HOW TO RUN (command line)
   python measure.py --calib ../outputs/calibration.npz \
                     --p1 610,420 --p2 980,430 --distance 2500
   (--distance in mm; --p1/--p2 are pixel coordinates of the two endpoints of
    the dimension you are measuring.)

 In practice you will usually pick the two endpoints in the web app (app.py),
 which calls measure_length() for you.
================================================================================
"""
import argparse
import numpy as np
import cv2
from utils import load_calibration


def undistort_pixels(points_xy, K, dist):
    """Map raw pixel points to undistorted pixel coordinates (same K)."""
    pts = np.asarray(points_xy, np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2)


def measure_length(p1, p2, Z_mm, K, dist):
    """
    Real-world length (mm) of the segment between image points p1 and p2,
    for an object facing the camera at depth Z_mm.
    Returns (length_mm, real_dx_mm, real_dy_mm).
    """
    (u1, v1), (u2, v2) = undistort_pixels([p1, p2], K, dist)
    fx, fy = K[0, 0], K[1, 1]
    real_dx = (u2 - u1) * Z_mm / fx
    real_dy = (v2 - v1) * Z_mm / fy
    length = float(np.hypot(real_dx, real_dy))
    return length, float(real_dx), float(real_dy)


def measure_bbox(x1, y1, x2, y2, Z_mm, K, dist):
    """Convenience: real width and height of an axis-aligned box (mm)."""
    width, _, _ = measure_length((x1, y1), (x2, y1), Z_mm, K, dist)
    height, _, _ = measure_length((x1, y1), (x1, y2), Z_mm, K, dist)
    return width, height


def _parse_xy(s):
    a, b = s.split(",")
    return (float(a), float(b))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 2: measure real object size")
    p.add_argument("--calib", default="../outputs/calibration.npz")
    p.add_argument("--p1", type=_parse_xy, required=True, help="x,y of endpoint 1 (px)")
    p.add_argument("--p2", type=_parse_xy, required=True, help="x,y of endpoint 2 (px)")
    p.add_argument("--distance", type=float, required=True, help="camera->object (mm)")
    a = p.parse_args()

    c = load_calibration(a.calib)
    length, dx, dy = measure_length(a.p1, a.p2, a.distance, c["camera_matrix"], c["dist_coeffs"])
    print(f"Distance (Z)     : {a.distance:.1f} mm")
    print(f"Pixel endpoints  : {a.p1} -> {a.p2}")
    print(f"Real dx, dy      : {dx:.1f} mm, {dy:.1f} mm")
    print(f"REAL LENGTH      : {length:.1f} mm  ({length/10:.2f} cm)")
