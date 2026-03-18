import streamlit as st
import numpy as np
import cv2
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image

st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter: Advanced Shape Designer")

# --- SIDEBAR: CHIP SHAPE DESIGNER ---
st.sidebar.header("📐 Chip Shape Designer")
st.sidebar.caption("Define the exact internal geometry of your chip.")

# Individual side controls
with st.sidebar.expander("✂️ Boundary Trims (px)", expanded=True):
    t_trim = st.slider("Top Trim (px)", 0, 800, 100)
    b_trim = st.slider("Bottom Trim (px)", 0, 800, 100)
    l_trim = st.slider("Left Trim (px)", 0, 800, 50)
    r_trim = st.slider("Right Trim (px)", 0, 800, 50)

# Individual corner controls
with st.sidebar.expander("🌀 Corner Rounding (px)", expanded=True):
    tl_rad = st.slider("Top-Left Radius (px)", 0, 400, 100)
    tr_rad = st.slider("Top-Right Radius (px)", 0, 400, 100)
    bl_rad = st.slider("Bottom-Left Radius (px)", 0, 400, 100)
    br_rad = st.slider("Bottom-Right Radius (px)", 0, 400, 100)

st.sidebar.markdown("---")

# Detection Settings
st.sidebar.header("⚙️ Detection Settings")
sensitivity = st.sidebar.slider("Sensitivity (x)", 0.1, 1.5, 0.5, 0.1, help="Multiplier for the brightness threshold. Lower = More sensitive.")
roundness = st.sidebar.slider("Roundness (Eccentricity)", 0.5, 1.0, 0.95, 0.05, help="0.0 = Circle, 1.0 = Line. Filters out elongated scratches.")

# Size Filters
st.sidebar.subheader("📏 Size Filters ($px^2$)")
min_area = st.sidebar.number_input("Min Area ($px^2$)", 1, 100, 15)
max_area = st.sidebar.number_input("Max Area ($px^2$)", 100, 5000, 800)

# --- FUNCTIONS ---

def create_custom_mask(h, w, t, b, l, r, tl, tr, bl, br):
    """ Generates a mask with 4 independent trims and 4 independent corner radii """
    mask = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = t, h - b
    x1, x2 = l, w - r
    
    # Ensure inner box is valid
    if y2 <= y1 or x2 <= x1: return mask.astype(bool)

    # Drawing the complex shape
    # 1. Main rectangles (cross shape to fill center)
    max_r = max(tl, tr, bl, br)
    cv2.rectangle(mask, (x1 + max_r, y1), (x2 - max_r, y2), 1, -1)
    cv2.rectangle(mask, (x1, y1 + max_r), (x2, y2 - max_r), 1, -1)
    
    # 2. Variable corners
    if tl > 0: cv2.circle(mask, (x1 + tl, y1 + tl), tl, 1, -1)
    else: cv2.rectangle(mask, (x1, y1), (x1+max_r, y1+max_r), 1, -1)
    
    if tr > 0: cv2.circle(mask, (x2 - tr, y1 + tr), tr, 1, -1)
    else: cv2.rectangle(mask, (x2-max_r, y1), (x2, y1+max_r), 1, -1)
    
    if bl > 0: cv2.circle(mask, (x1 + bl, y2 - bl), bl, 1, -1)
    else: cv2.rectangle(mask, (x1, y2-max_r), (x1+max_r, y2), 1, -1)
    
    if br > 0: cv2.circle(mask, (x2 - br, y2 - br), br, 1, -1)
    else: cv2.rectangle(mask, (x2-max_r, y2-max_r), (x2, y2), 1, -1)
    
    return mask.astype(bool)

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

# --- MAIN APP UI ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    raw_img = np.array(Image.open(uploaded_file))
    if len(raw_img.shape) == 3:
        gray_img = color.rgb2gray(raw_img[:, :, :3])
    else:
        gray_img = raw_img
    
    h, w = gray_img.shape
    
    # 1. Update Mask (Instant)
    analysis_mask = create_custom_mask(h, w, t_trim, b_trim, l_trim, r_trim, tl_rad, tr_rad, bl_rad, br_rad)
    
    # 2. Red Overlay Preview
    overlay = (gray_img * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    red_shroud = np.zeros_like(overlay)
    red_shroud[:] = [255, 0, 0]
    overlay[~analysis_mask] = cv2.addWeighted(overlay[~analysis_mask], 0.7, red_shroud[~analysis_mask], 0.3, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: Align Mask")
        st.image(overlay, use_column_width=True)
        run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    with col2:
        st.subheader("Step 2: Results")
        if run_btn:
            with st.spinner("Counting..."):
                cells = analyze_cells(gray_img, analysis_mask, sensitivity, roundness, min_area, max_area)
                res_disp = (gray_img * 255).astype(np.uint8)
                res_disp = cv2.cvtColor(res_disp, cv2.COLOR_GRAY2RGB)
                for c in cells:
                    y, x = c.centroid
                    rad = int(c.equivalent_diameter_area / 2 * 1.2)
                    cv2.circle(res_disp, (int(x), int(y)), rad, (0, 255, 0), 2)
                
                st.image(res_disp, use_column_width=True)
                st.metric("Cell Count", len(cells))
        else:
            st.info("Design your mask on the left, then click 'Run Analysis'.")
