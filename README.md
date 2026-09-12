# CSc 8830 Computer Vision — Module 2
## Camera Calibration & Real-World Object Measurement

A web application that (1) calibrates a smartphone camera from checkerboard
photos, (2) measures the real-world 2D size of an object from a single photo
using perspective projection, and (3) validates the method on 20 measurements
taken from > 2 m away and reports error statistics.

---

## Folder structure
```
Module2_CameraCalibration/
├── app.py                  # Streamlit web app (all parts accessible here)
├── requirements.txt
├── measurements_log.csv    # your 20 validation measurements go here
├── src/
│   ├── calibrate.py        # Step 1: camera calibration
│   ├── measure.py          # Step 2: perspective-projection measurement
│   ├── validate.py         # Step 3: validation + error statistics
│   ├── synthetic.py        # generates known-truth test data (dev only)
│   └── utils.py            # save/load calibration
├── data/
│   ├── calibration_images/ # <-- put your ~25 checkerboard phone photos here
│   ├── validation_images/  # <-- put your 20 object photos here
│   └── synthetic/          # auto-generated test images
└── outputs/                # calibration + validation results
```

## Setup
```
pip install -r requirements.txt
```

## Run the web application (this is what you screen-record)
```
streamlit run app.py
```
Then open http://localhost:8501 and use the tabs:
Overview · Step 1 Calibration · Step 2 Measure · Step 3 Validation · Theory.

## Run from the command line (optional)
```
# Step 1 — calibrate (set --cols/--rows to your board's INNER corners)
python src/calibrate.py --images data/calibration_images \
    --cols 9 --rows 6 --square_size 24.0 --out outputs/calibration.npz --debug

# Step 2 — measure one object (pixel endpoints of the dimension, distance in mm)
python src/measure.py --calib outputs/calibration.npz \
    --p1 610,420 --p2 980,430 --distance 2500

# Step 3 — validation + error stats (after filling measurements_log.csv)
python src/validate.py --calib outputs/calibration.npz \
    --log measurements_log.csv --out outputs/results
```

## Assignment workflow
1. Print/keep a checkerboard, tape it flat, measure ONE square (mm), count the
   INNER corners (e.g. 9×6).
2. Take ~25 varied phone photos (different tilts/angles/distances, whole board
   visible, 1× zoom, same resolution) → `data/calibration_images/`.
3. Run Step 1 → get the camera matrix.
4. For validation: photograph objects from > 2 m (tape-measure the distance AND
   the true object size). In the app, click the two endpoints of each dimension,
   enter the true size + distance, and log it. Collect 20 measurements.
5. Run Step 3 → error statistics + plots.
6. Record a screen capture of the web app performing Steps 1–3.

## Development note (synthetic data)
`src/synthetic.py` renders checkerboards and objects from a *virtual* camera
with known intrinsics, to prove the code is correct before real photos exist
(recovered fx,fy ≈ 1401 vs true 1400; validation error ≈ 0.1%). The **submitted
calibration must use real smartphone photos** — synthetic images only test the
code, they cannot recover your phone's real lens.
