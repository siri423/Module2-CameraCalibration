"""
================================================================================
 app.py  --  Web application for CSc 8830 Module 2 (Streamlit)
================================================================================
 The assignment requires all parts to be accessible via one webpage. This app
 exposes: Overview, Step 1 (Calibration), Step 2 (Measure an object),
 Step 3 (Validation + error stats), and the Theory derivation.

 RUN:  streamlit run app.py
       then open the URL it prints (default http://localhost:8501)
================================================================================
"""
import os
import sys
import csv
import numpy as np
from PIL import Image
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
from calibrate import calibrate                         # noqa: E402
from measure import measure_length                       # noqa: E402
from validate import run as run_validation               # noqa: E402
from utils import load_calibration                        # noqa: E402

CAL_IMG_DIR = os.path.join(HERE, "data", "calibration_images")
VAL_IMG_DIR = os.path.join(HERE, "data", "validation_images")
OUT = os.path.join(HERE, "outputs")
CALIB_PATH = os.path.join(OUT, "calibration.npz")
LOG_PATH = os.path.join(HERE, "measurements_log.csv")
RESULTS = os.path.join(OUT, "results")
LOG_HEADER = ["image", "object", "dimension", "true_size_mm",
              "distance_mm", "x1", "y1", "x2", "y2"]

st.set_page_config(page_title="CSc 8830 Module 2 - Camera Calibration", layout="wide")
st.title("CSc 8830 - Module 2: Camera Calibration & Object Measurement")

tabs = st.tabs(["Overview", "Step 1 - Calibration", "Step 2 - Measure Object",
                "Step 3 - Validation", "Theory"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.header("Overview")
    st.markdown("""
This web application implements the full Module 2 pipeline:

1. **Calibrate** a smartphone camera from checkerboard photos (recover the
   intrinsic matrix **K** and lens distortion).
2. **Measure** the real-world 2D size of an object from a single photo using
   the perspective-projection equations.
3. **Validate** on 20 measurements taken from > 2 m away and report error
   statistics.

Use the tabs above. Run **Step 1** first; Steps 2 and 3 use its calibration.
""")
    if os.path.exists(CALIB_PATH):
        c = load_calibration(CALIB_PATH)
        st.success(f"Calibration loaded  -  fx={c['camera_matrix'][0,0]:.1f}, "
                   f"fy={c['camera_matrix'][1,1]:.1f}, RMS={c['rms']:.3f}px")
    else:
        st.info("No calibration yet - run Step 1.")

# ------------------------------------------------------------------ Step 1
with tabs[1]:
    st.header("Step 1 - Camera Calibration")
    st.markdown("Upload your checkerboard photos, set the pattern, and run.")
    col1, col2, col3 = st.columns(3)
    cols = col1.number_input("Inner corners across (cols)", 3, 30, 9)
    rows = col2.number_input("Inner corners down (rows)", 3, 30, 6)
    square = col3.number_input("Square size (mm)", 1.0, 200.0, 25.0, step=0.5)

    ups = st.file_uploader("Checkerboard images", type=["jpg", "jpeg", "png"],
                           accept_multiple_files=True)
    use_syn = st.checkbox("Use synthetic test images instead (data/synthetic/calibration)")

    if st.button("Run calibration", type="primary"):
        img_dir = os.path.join(HERE, "data", "synthetic", "calibration") if use_syn else CAL_IMG_DIR
        if not use_syn:
            os.makedirs(CAL_IMG_DIR, exist_ok=True)
            for u in ups:
                with open(os.path.join(CAL_IMG_DIR, u.name), "wb") as f:
                    f.write(u.getbuffer())
        try:
            with st.spinner("Detecting corners and calibrating..."):
                K, dist, size, rms = calibrate(img_dir, (int(cols), int(rows)),
                                               float(square), CALIB_PATH)
            st.success(f"Done. RMS reprojection error = {rms:.3f} px")
            st.write("**Camera matrix K**"); st.dataframe(np.round(K, 2))
            st.write(f"**Distortion:** {np.round(dist.ravel(), 4).tolist()}")
            st.write(f"**Image size:** {size[0]} x {size[1]}")
            with open(os.path.splitext(CALIB_PATH)[0] + ".json") as f:
                st.download_button("Download calibration.json", f.read(),
                                   "calibration.json")
        except SystemExit as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ Step 2
with tabs[2]:
    st.header("Step 2 - Measure a Real Object")
    if not os.path.exists(CALIB_PATH):
        st.warning("Run Step 1 first.")
    else:
        c = load_calibration(CALIB_PATH)
        K, dist = c["camera_matrix"], c["dist_coeffs"]
        img_up = st.file_uploader("Object photo", type=["jpg", "jpeg", "png"], key="obj")
        dist_mm = st.number_input("Camera-to-object distance Z (mm)", 100.0,
                                  20000.0, 2500.0, step=10.0)
        if img_up:
            pil = Image.open(img_up).convert("RGB")
            ow, oh = pil.size
            disp_w = min(720, ow)
            scale = ow / disp_w
            if "pts" not in st.session_state:
                st.session_state.pts = []
            st.caption("Click the TWO endpoints of the dimension you want "
                       "(e.g. left & right edge for width).")
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                disp = pil.resize((disp_w, int(oh / scale)))
                v = streamlit_image_coordinates(disp, key="clicker")
                if v is not None:
                    p = (round(v["x"] * scale, 1), round(v["y"] * scale, 1))
                    if not st.session_state.pts or st.session_state.pts[-1] != p:
                        st.session_state.pts.append(p)
                        st.session_state.pts = st.session_state.pts[-2:]
            except Exception:
                st.info("Click component unavailable - enter pixel coords below.")
            cc = st.columns(4)
            if cc[0].button("Reset points"):
                st.session_state.pts = []
            with st.expander("Or enter pixel coordinates manually"):
                mc = st.columns(4)
                x1 = mc[0].number_input("x1", 0.0, float(ow), 0.0)
                y1 = mc[1].number_input("y1", 0.0, float(oh), 0.0)
                x2 = mc[2].number_input("x2", 0.0, float(ow), 0.0)
                y2 = mc[3].number_input("y2", 0.0, float(oh), 0.0)
                if st.button("Use these coordinates"):
                    st.session_state.pts = [(x1, y1), (x2, y2)]
            st.write("Selected points:", st.session_state.pts)
            if len(st.session_state.pts) == 2:
                p1, p2 = st.session_state.pts
                length, dx, dy = measure_length(p1, p2, dist_mm, K, dist)
                st.metric("Estimated real length", f"{length:.1f} mm  ({length/10:.2f} cm)")
                st.caption(f"real dx={dx:.1f} mm, dy={dy:.1f} mm")
                st.divider()
                st.markdown("**Log this as a validation measurement (Step 3):**")
                lc = st.columns(3)
                obj = lc[0].text_input("Object name", "object")
                dim = lc[1].selectbox("Dimension", ["width", "height", "length"])
                true = lc[2].number_input("True size (mm, tape-measured)", 0.0,
                                          5000.0, 0.0)
                if st.button("Add to measurement log"):
                    new = not os.path.exists(LOG_PATH)
                    with open(LOG_PATH, "a", newline="") as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(LOG_HEADER)
                        w.writerow([img_up.name, obj, dim, true, dist_mm,
                                    p1[0], p1[1], p2[0], p2[1]])
                    st.success("Logged. See Step 3.")

# ------------------------------------------------------------------ Step 3
with tabs[3]:
    st.header("Step 3 - Validation & Error Statistics")
    if os.path.exists(LOG_PATH):
        import pandas as pd
        df = pd.read_csv(LOG_PATH)
        st.write(f"Measurements logged: **{len(df)}** (need 20)")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No measurements logged yet. Use Step 2, or point --log at a CSV.")
    use_syn_v = st.checkbox("Validate on synthetic log instead")
    if st.button("Run validation", type="primary"):
        log = os.path.join(OUT, "synthetic_measurements_log.csv") if use_syn_v else LOG_PATH
        try:
            rows, stats = run_validation(CALIB_PATH, log, RESULTS)
            st.text(stats)
            plot = os.path.join(RESULTS, "error_plots.png")
            if os.path.exists(plot):
                st.image(plot)
            import pandas as pd
            st.dataframe(pd.read_csv(os.path.join(RESULTS, "validation_results.csv")),
                         use_container_width=True)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ Theory
with tabs[4]:
    st.header("Theory - Two-camera relationship of a point P(X, Y, Z)")
    st.markdown("Camera 1 is static (world = camera-1 frame). Camera 2 is at "
                "rotation **R** and translation **t** (oblique, offset).")
    st.latex(r"\lambda_1 \begin{bmatrix} u_1 \\ v_1 \\ 1 \end{bmatrix} = K_1 \, X_1,"
             r"\qquad X_1 = (X, Y, Z)^T")
    st.latex(r"X_2 = R\,X_1 + t")
    st.latex(r"\lambda_2 \begin{bmatrix} u_2 \\ v_2 \\ 1 \end{bmatrix} = K_2 \, X_2 = K_2 (R\,X_1 + t)")
    st.markdown("Eliminating the 3D point gives the epipolar (essential/fundamental) relation:")
    st.latex(r"x_2^{\top} E \, x_1 = 0,\quad E = [t]_\times R,\qquad "
             r"p_2^{\top} F \, p_1 = 0,\quad F = K_2^{-\top}[t]_\times R\,K_1^{-1}")
    st.markdown("Explicit forward map for a point at depth $Z_1$ in camera 1:")
    st.latex(r"p_2 = \frac{K_2\left(R\,Z_1 K_1^{-1} p_1 + t\right)}"
             r"{e_3^{\top}\left(R\,Z_1 K_1^{-1} p_1 + t\right)}")
    st.markdown(r"""
**Static parameters** $K_1,K_2$ - from calibrating each camera (Step 1).
$R,t$ - relative pose from stereo calibration or a known rig.
**Variables** $p_1,p_2$ (image points) and depth $Z_1$ (scene structure).
**Assumptions:** ideal pinhole model, known/estimated intrinsics, a rigid
relative pose between cameras, and P inside both fields of view.
""")
