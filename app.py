import streamlit as st
import numpy as np
import cv2
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter: Research Edition")

# --- SIDEBAR: DOCUMENTATION & GUIDE ---
with st.sidebar:
    st.header("📖 Quick Guide")
    with st.expander("How to use this tool"):
        st.write("""
        1. **Upload** your microscopy image.
        2. **Align the Mask** using the Trims and Corner sliders. The **Red Area** will be ignored.
        3. **Set Sensitivity** based on cell brightness.
        4. **Filter by Size** to remove dust or large debris.
        5. Click **Run Analysis** to see results.
        """)
    st.markdown("---")

# --- SIDEBAR: CHIP SHAPE DESIGNER ---
st.sidebar.header("📐 Chip Shape Designer")

with st.sidebar.expander("✂️ Boundary Trims (px)", expanded=True):
    t_trim = st.slider("Top Trim (px)", 0, 2000, 100, help="Removes rows from the top of the image. Useful for ignoring the top chip wall.")
    b_trim = st.slider("Bottom Trim (px)", 0, 2000, 100, help="Removes rows from the bottom. Useful for ignoring scale bars or bottom walls.")
    l_trim = st.slider("Left Trim (px)", 0, 2000, 50, help="Removes columns from the left side.")
    r_trim = st.slider("Right Trim (px)", 0, 2000, 50, help="Removes columns from the right side.")

with st.sidebar.expander("🌀 Corner Rounding (px)", expanded=True):
    tl_rad = st.slider("Top-Left Radius (px)", 0, 3000, 100, help="Curves the top-left corner. Set to half the channel height for a perfect semicircle.")
    tr_rad = st.slider("Top-Right Radius (px)", 0, 3000, 100, help="Curves the top-right corner.")
    bl_rad = st.slider("Bottom-Left Radius (px)", 0, 3000, 100, help="Curves the bottom-left corner.")
    br_rad = st.slider("Bottom-Right Radius (px)", 0, 3000, 100, help="Curves the bottom-right corner.")

st.sidebar.markdown("---")

# --- SIDEBAR: DETECTION SETTINGS ---
st.sidebar.header("⚙️ Detection Settings")

sensitivity = st.sidebar.slider(
    "Sensitivity (x)", 0.1, 1.5, 0.5, 0.1, 
    help="Multiplier for the brightness threshold. 0.1 is MOST sensitive (catches dim cells); 1.5 is LEAST sensitive (only brightest cells)."
)

roundness = st.sidebar.slider(
    "Roundness (Eccentricity)", 0.5, 1.0, 0.95, 0.05, 
    help="Filters objects by shape. 0.0 is a perfect circle; 1.0 is a line. Lowering this value removes long scratches."
)

st.sidebar.subheader("📏 Size Filters ($px^2$)")
min_area = st.sidebar.number_input(
    "Min Area ($px^2$)", 1, 500, 15, 
    help="The 'Dust Filter.' Any object smaller than this total pixel area is ignored."
)
max_area = st.sidebar.number_input(
    "Max Area ($px^2$)", 100, 10000, 800, 
    help="The 'Clump Filter.' Any object larger than this total pixel area (like scratches or cell clusters) is ignored."
)

# --- PROCESSING FUNCTIONS ---

def create_custom_mask(h, w, t, b, l, r, tl, tr, bl, br):
    mask = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = t, h - b
    x1, x2 = l, w - r
    if y2 <= y1 or x2 <= x1: return mask.astype(bool)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    corner_mask = np.ones((h, w), dtype=np.uint8)
    if tl > 0:
        cv2.rectangle(corner_mask, (x1, y1), (x1+tl, y1+tl), 0, -1)
        cv2.circle(corner_mask, (x1+tl, y1+tl), tl, 1, -1)
    if tr > 0:
        cv2.rectangle(corner_mask, (x2-tr, y1), (x2, y1+tr), 0, -1)
        cv2.circle(corner_mask, (x2-tr, y1+tr), tr, 1, -1)
    if bl > 0:
        cv2.rectangle(corner_mask, (x1, y2-bl), (x1+bl, y2), 0, -1)
        cv2.circle(corner_mask, (x1+bl, y2-bl), bl, 1, -1)
    if br > 0:
        cv2.rectangle(corner_mask, (x2-br, y2-br), (x2, y2), 0, -1)
        cv2.circle(corner_mask, (x2-br, y2-br), br, 1, -1)
    return cv2.bitwise_and(mask, corner_mask).astype(bool)

@st.cache_data
def analyze_cells(img, mask, sens, round_val, min_a, max_a):
    clean = np.copy(img)
    clean[-50:, :] = 0
    top_hat = morphology.white_tophat(clean, morphology.disk(15))
    bw = (top_hat > (filters.threshold_otsu(top_hat) * sens)) & mask
    dist = ndi.distance_transform_edt(bw)
    peaks = feature.peak_local_max(dist, min_distance=4, labels=bw)
    m = np.zeros(dist.shape, dtype=bool)
    if len(peaks) > 0: m[tuple(peaks.T)] = True
    markers, _ = ndi.label(m)
    labels = segmentation.watershed(-dist, markers, mask=bw)
    regions = measure.regionprops(labels)
    valid = [r for r in regions if min_a <= r.area <= max_a and r.eccentricity < round_val]
    return valid

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Microscopy Image", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    raw_img = np.array(Image.open(uploaded_file))
    gray_img = color.rgb2gray(raw_img[:, :, :3]) if len(raw_img.shape) == 3 else raw_img
    h, w = gray_img.shape
    
    # Instant Mask Preview
    analysis_mask = create_custom_mask(h, w, t_trim, b_trim, l_trim, r_trim, tl_rad, tr_rad, bl_rad, br_rad)
    overlay = (gray_img * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    red_shroud = np.zeros_like(overlay)
    red_shroud[:] = [255, 0, 0]
    overlay[~analysis_mask] = cv2.addWeighted(overlay[~analysis_mask], 0.7, red_shroud[~analysis_mask], 0.3, 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Design Chip Mask")
        st.image(overlay, use_column_width=True)
        run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    with col2:
        st.subheader("2. Analysis Results")
        if run_btn:
            with st.spinner("Executing Watershed Algorithm..."):
                cells = analyze_cells(gray_img, analysis_mask, sensitivity, roundness, min_area, max_area)
                res_disp = (gray_img * 255).astype(np.uint8)
                res_disp = cv2.cvtColor(res_disp, cv2.COLOR_GRAY2RGB)
                
                data = []
                for c in cells:
                    y, x = c.centroid
                    rad = int(c.equivalent_diameter_area / 2 * 1.2)
                    cv2.circle(res_disp, (int(x), int(y)), rad, (0, 255, 0), 2)
                    data.append({"Cell_ID": len(data)+1, "X": x, "Y": y, "Area_px2": c.area})
                
                st.image(res_disp, use_column_width=True)
                st.metric("Total Cells Counted", len(cells))
                
                if data:
                    df = pd.DataFrame(data)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results (CSV)", csv, f"count_{len(cells)}.csv", "text/csv")
        else:
            st.info("Align the red shroud with your chip boundaries, then click 'Run Analysis'.")
