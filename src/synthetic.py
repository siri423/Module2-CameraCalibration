"""
================================================================================
 synthetic.py  --  Generate synthetic test data with KNOWN ground truth
================================================================================
 CSc 8830 Computer Vision, Module 2.

 PURPOSE (development only -- NOT the submitted calibration)
   Renders checkerboard images from a virtual camera whose true intrinsics we
   know, plus rectangle "object" images at known sizes and distances. This lets
   us prove calibrate.py / measure.py / validate.py are correct BEFORE the real
   smartphone photos exist: calibration should recover ~the true camera matrix,
   and measured sizes should match the true sizes.

   The FINAL submission must use REAL smartphone photos (assignment Step 1),
   because calibration must recover YOUR phone's real lens -- synthetic images
   only validate the code logic.

 HOW TO RUN
   python synthetic.py
   -> writes ../data/synthetic/calibration/*.png
             ../data/synthetic/objects/*.png
             ../outputs/synthetic_measurements_log.csv
             ../outputs/synthetic_truth.json
================================================================================
"""
import json
import os
import numpy as np
import cv2

# ---- virtual camera ground truth ----
W, H = 1280, 960
FX = FY = 1400.0
CX, CY = W / 2.0, H / 2.0
K_TRUE = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float64)
DIST_TRUE = np.array([-0.08, 0.02, 0.0, 0.0, 0.0], np.float64)   # mild radial

# ---- checkerboard ground truth ----
COLS, ROWS = 9, 6            # inner corners (must match calibrate --cols/--rows)
SQUARE = 25.0               # mm
NCOLS_SQ, NROWS_SQ = COLS + 1, ROWS + 1

HERE = os.path.dirname(__file__)
CAL_DIR = os.path.join(HERE, "..", "data", "synthetic", "calibration")
OBJ_DIR = os.path.join(HERE, "..", "data", "synthetic", "objects")
OUT_DIR = os.path.join(HERE, "..", "outputs")


def euler_to_rvec(rx, ry, rz):
    """Euler angles (radians) -> Rodrigues rotation vector."""
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rvec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
    return rvec


def board_black_squares():
    """3D corners (centred, z=0, mm) of every BLACK square on the board."""
    cx = NCOLS_SQ * SQUARE / 2.0
    cy = NROWS_SQ * SQUARE / 2.0
    squares = []
    for a in range(NCOLS_SQ):
        for b in range(NROWS_SQ):
            if (a + b) % 2 == 0:                       # black square
                x0, y0 = a * SQUARE - cx, b * SQUARE - cy
                x1, y1 = x0 + SQUARE, y0 + SQUARE
                squares.append(np.array([[x0, y0, 0], [x1, y0, 0],
                                         [x1, y1, 0], [x0, y1, 0]], np.float64))
    return squares


def render_board(rvec, tvec):
    img = np.full((H, W, 3), 255, np.uint8)            # white background = margin
    for sq in board_black_squares():
        pts, _ = cv2.projectPoints(sq, rvec, tvec, K_TRUE, DIST_TRUE)
        cv2.fillConvexPoly(img, np.round(pts).astype(np.int32), (10, 10, 10))
    return img


def all_inner_corners_visible(rvec, tvec, margin=40):
    cx = NCOLS_SQ * SQUARE / 2.0
    cy = NROWS_SQ * SQUARE / 2.0
    pts3d = []
    for i in range(COLS):
        for j in range(ROWS):
            pts3d.append([(i + 1) * SQUARE - cx, (j + 1) * SQUARE - cy, 0])
    pts, _ = cv2.projectPoints(np.array(pts3d, np.float64), rvec, tvec, K_TRUE, DIST_TRUE)
    p = pts.reshape(-1, 2)
    return (p[:, 0].min() > margin and p[:, 0].max() < W - margin and
            p[:, 1].min() > margin and p[:, 1].max() < H - margin)


def make_calibration_images(n=22, seed=0):
    os.makedirs(CAL_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    made = 0
    tries = 0
    while made < n and tries < n * 40:
        tries += 1
        rx = np.radians(rng.uniform(-35, 35))
        ry = np.radians(rng.uniform(-35, 35))
        rz = np.radians(rng.uniform(-25, 25))
        Z = rng.uniform(480, 780)
        tx = rng.uniform(-60, 60)
        ty = rng.uniform(-45, 45)
        rvec = euler_to_rvec(rx, ry, rz)
        tvec = np.array([[tx], [ty], [Z]], np.float64)
        if not all_inner_corners_visible(rvec, tvec):
            continue
        img = render_board(rvec, tvec)
        cv2.imwrite(os.path.join(CAL_DIR, f"calib_{made:02d}.png"), img)
        made += 1
    print(f"Wrote {made} synthetic calibration images -> {CAL_DIR}")
    return made


def make_object_images(seed=1):
    """10 rectangles, log width AND height each = 20 measurements. Z > 2000 mm."""
    os.makedirs(OBJ_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = [("image", "object", "dimension", "true_size_mm", "distance_mm",
             "x1", "y1", "x2", "y2")]
    for k in range(10):
        Wmm = float(rng.uniform(120, 400))
        Hmm = float(rng.uniform(120, 400))
        Z = float(rng.uniform(2100, 4200))
        # object plane facing camera at depth Z, centred on the optical axis
        corners = np.array([[-Wmm / 2, -Hmm / 2, Z], [Wmm / 2, -Hmm / 2, Z],
                            [Wmm / 2, Hmm / 2, Z], [-Wmm / 2, Hmm / 2, Z]], np.float64)
        pts, _ = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), K_TRUE, DIST_TRUE)
        p = pts.reshape(-1, 2)
        img = np.full((H, W, 3), 240, np.uint8)
        cv2.fillConvexPoly(img, np.round(p).astype(np.int32), (60, 90, 160))
        cv2.polylines(img, [np.round(p).astype(np.int32)], True, (0, 0, 0), 2)
        name = f"obj_{k:02d}.png"
        cv2.imwrite(os.path.join(OBJ_DIR, name), img)
        tl, tr, br, bl = p
        # width = top edge (tl->tr); height = left edge (tl->bl)
        rows.append((name, f"rect{k}", "width", round(Wmm, 2), round(Z, 1),
                     round(tl[0], 2), round(tl[1], 2), round(tr[0], 2), round(tr[1], 2)))
        rows.append((name, f"rect{k}", "height", round(Hmm, 2), round(Z, 1),
                     round(tl[0], 2), round(tl[1], 2), round(bl[0], 2), round(bl[1], 2)))
    log = os.path.join(OUT_DIR, "synthetic_measurements_log.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(log, "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Wrote {len(rows)-1} synthetic measurements -> {log}")


def main():
    make_calibration_images()
    make_object_images()
    with open(os.path.join(OUT_DIR, "synthetic_truth.json"), "w") as f:
        json.dump({"K_true": K_TRUE.tolist(), "dist_true": DIST_TRUE.tolist(),
                   "square_size_mm": SQUARE, "pattern_inner_corners": [COLS, ROWS],
                   "image_size": [W, H]}, f, indent=2)
    print("Ground truth saved -> outputs/synthetic_truth.json")


if __name__ == "__main__":
    main()
