"""
================================================================================
 calibrate.py  --  STEP 1: Camera calibration from checkerboard photos
================================================================================
 CSc 8830 Computer Vision, Module 2.

 WHAT IT DOES
   Reads all checkerboard photos in a folder, detects the inner corners of the
   pattern in each, and runs OpenCV's calibrateCamera to recover the camera's
   intrinsic matrix K (focal lengths fx,fy and optical center cx,cy) and the
   lens distortion coefficients. Results are saved to outputs/calibration.npz
   (+ a readable .json).

 HOW TO RUN
   python calibrate.py --images ../data/calibration_images \
                       --cols 9 --rows 6 --square_size 24.0 \
                       --out ../outputs/calibration.npz --debug

   --cols / --rows : number of INNER corners (where 4 squares meet), NOT squares.
                     A board that is 10x7 squares has 9x6 inner corners.
   --square_size   : side length of one square in millimetres (measure it once).
   --debug         : also saves images with detected corners drawn, so you can
                     visually confirm detection worked.

 NOTE: cols/rows and square_size are the ONLY physical inputs. No per-photo
 distance or angle is needed -- OpenCV recovers each board's pose automatically.
================================================================================
"""
import argparse
import glob
import os
import cv2
import numpy as np
from utils import save_calibration

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def find_corners(gray, pattern_size):
    """Try the modern SB detector first, fall back to the classic one."""
    # findChessboardCornersSB is more robust to blur/lighting (OpenCV >= 4.0)
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            return True, corners
    except Exception:
        pass
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def calibrate(image_dir, pattern_size, square_size_mm, out_path, debug=False):
    cols, rows = pattern_size

    # 3D coordinates of the inner corners on the (flat) board, in millimetres.
    # z = 0 because the board is planar. Scaled by the real square size so the
    # recovered translations come out in millimetres.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points, img_points = [], []      # 3D board points / 2D image points
    used, skipped = [], []
    image_size = None

    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No images found in {image_dir}")

    dbg_dir = os.path.join(os.path.dirname(out_path) or ".", "results", "corners")
    if debug:
        os.makedirs(dbg_dir, exist_ok=True)

    for f in files:
        img = cv2.imread(f)
        if img is None:
            skipped.append((os.path.basename(f), "unreadable"))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])   # (w, h)
        found, corners = find_corners(gray, pattern_size)
        if found:
            obj_points.append(objp.copy())
            img_points.append(corners)
            used.append(os.path.basename(f))
            if debug:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                cv2.imwrite(os.path.join(dbg_dir, os.path.basename(f)), vis)
        else:
            skipped.append((os.path.basename(f), "pattern not found"))

    print(f"\nDetected the {cols}x{rows} pattern in {len(used)}/{len(files)} images.")
    for name, why in skipped:
        print(f"   skipped: {name}  ({why})")
    if len(used) < 5:
        raise SystemExit("Need the pattern detected in at least ~5 images "
                         "(10-15 recommended). Check --cols/--rows and photo quality.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    # Per-image reprojection error (a quality check; lower is better, <1px is good)
    sq_err, n_pts = 0.0, 0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        a = img_points[i].reshape(-1, 2).astype(np.float64)
        b = proj.reshape(-1, 2).astype(np.float64)
        sq_err += float(np.sum((a - b) ** 2))
        n_pts += len(a)
    mean_err = np.sqrt(sq_err / n_pts)

    save_calibration(out_path, K, dist, image_size, rms, square_size_mm, pattern_size)

    print("\n================ CALIBRATION RESULT ================")
    print(f"Images used            : {len(used)}")
    print(f"Image size (w x h)     : {image_size[0]} x {image_size[1]}")
    print(f"RMS reprojection error : {rms:.4f} px   (lower is better; <1 is good)")
    print(f"Mean reprojection error: {mean_err:.4f} px")
    print(f"fx, fy                 : {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"cx, cy                 : {K[0,2]:.2f}, {K[1,2]:.2f}")
    print(f"Distortion coeffs      : {dist.ravel()}")
    print(f"\nSaved -> {out_path} (+ .json)")
    return K, dist, image_size, rms


def build_argparser():
    p = argparse.ArgumentParser(description="Step 1: camera calibration")
    p.add_argument("--images", required=True, help="folder of checkerboard photos")
    p.add_argument("--cols", type=int, required=True, help="inner corners across")
    p.add_argument("--rows", type=int, required=True, help="inner corners down")
    p.add_argument("--square_size", type=float, required=True, help="square side (mm)")
    p.add_argument("--out", default="../outputs/calibration.npz")
    p.add_argument("--debug", action="store_true", help="save drawn-corner images")
    return p


if __name__ == "__main__":
    a = build_argparser().parse_args()
    calibrate(a.images, (a.cols, a.rows), a.square_size, a.out, a.debug)
