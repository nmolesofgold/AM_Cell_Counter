import streamlit as st
import numpy as np
import cv2
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image

st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter: Chip Shape Designer")

# --- SIDEBAR: BOUNDARY DESIGNER ---
st.sidebar.header("📐 Chip Shape Designer")
st.sidebar.caption("Define the internal area of your chip to ignore the walls.")

# Individual side controls
t_trim = st.sidebar.slider("Top Trim", 0, 500, 50)
b_trim = st.sidebar.slider("Bottom Trim", 0, 500, 50)
l_trim = st.sidebar.slider("Left Trim", 0, 500, 20)
r_trim = st.sidebar.slider("Right Trim", 0, 500, 20)

# Corner Rounding (The Semicircle Tool)
corner_radius = st.sidebar.slider(
    "Corner Rounding (Radius)", 0, 250, 50, 
    help="Increase this to match the semicircular ends of your chip."
)

st.sidebar.header("⚙️ Detection Settings")
sensitivity = st.sidebar.slider("Sensitivity", 0.1, 1.5, 0.5, 0.1)
roundness = st.sidebar.slider("Roundness Tolerance", 0.5, 1.0, 0.95, 0.05)

st.sidebar.subheader("Size Filters")
min_area = st.sidebar.number_input("Min Size (px)", 1, 100, 15)
max_area = st.sidebar.number_input("Max Size (px)", 100, 5000, 800)

@st.cache_data
def create_rounded_mask(h, w, t, b, l, r, radius):
    """ Generates a mask with 4 independent sides and rounded corners """
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define the rectangle inner bounds
    y1, y2 = t, h - b
    x1, x2 = l, w - r
    
    # Ensure radius isn't larger than the box itself
    radius = min(radius, (y2 - y1) // 2, (x2 - x1) // 2)
    
    # Draw the rounded rectangle using OpenCV (much faster than manual loops)
    # 1. Draw the central horizontal and vertical bars
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 1, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 1, -1)
    
    # 2. Draw the 4 corner circles
    if radius > 0:
        cv2.circle(mask, (x1 + radius, y1 + radius), radius, 1, -1) # Top-Left
        cv2.circle(mask, (x2 - radius, y1 + radius), radius, 1, -1) # Top-Right
        cv2.circle(mask, (x1 + radius, y2 - radius), radius, 1, -1) # Bottom-Left
        cv2.circle(mask, (x2 - radius, y2 - radius), radius, 1, -1) # Bottom-Right
        
    return mask.astype(bool)

@st.cache_data
def run_analysis(image_array, sens, round_val, min_a, max_a, t, b, l, r, radius):
    if len(image_array.shape) == 3:
        img = color.rgb2gray(image_array[:, :, :3])
    else:
        img = image_array
        
    h, w = img.shape
    analysis_mask = create_rounded_mask(h, w, t, b, l, r, radius)
    
    # Pre-processing
    clean = np.copy(img)
    clean[-50:, :] = 0 # Scale bar wipe
    top_hat = morphology.white_tophat(clean, morphology.disk(15))
    
    # Thresholding + Masking
    bw = (top_hat > (filters.threshold_otsu(top_hat) * sens)) & analysis_mask

    # Watershed
    dist = ndi.distance_transform_edt(bw)
    peaks = feature.peak_local_max(dist, min_distance=4, labels=bw)
    m = np.zeros(dist.shape, dtype=bool)
    if len(peaks) > 0: m[tuple(peaks.T)] = True
    markers, _ = ndi.label(m)
    labels = segmentation.watershed(-dist, markers, mask=bw)

    regions = measure.regionprops(labels)
    valid = [r for r in regions if min_a <= r.area <= max_a and r.eccentricity < round_val]
    
    return img, valid, analysis_mask

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "tif"])

if uploaded_file:
    raw_img = np.array(Image.open(uploaded_file))
    
    # Run heavy math
    gray_img, cells, analysis_mask = run_analysis(
        raw_img, sensitivity, roundness, min_area, max_area, 
        t_trim, b_trim, l_trim, r_trim, corner_radius
    )
    
    # --- VISUAL PREPARATION ---
    # 1. Boundary Preview (Original image dimmed outside the mask)
    orig_disp = (gray_img * 255).astype(np.uint8)
    orig_disp = cv2.cvtColor(orig_disp, cv2.COLOR_GRAY2RGB)
    orig_disp[~analysis_mask] = (orig_disp[~analysis_mask] * 0.3).astype(np.uint8)
    
    # 2. Results Preview
    res_disp = (gray_img * 255).astype(np.uint8)
    res_disp = cv2.cvtColor(res_disp, cv2.COLOR_GRAY2RGB)
    for c in cells:
        y, x = c.centroid
        rad = int(c.equivalent_diameter_area / 2 * 1.2)
        cv2.circle(res_disp, (int(x), int(y)), rad, (255, 0, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Chip Mask Designer")
        st.image(orig_disp, use_column_width=True)
    with col2:
        st.subheader(f"2. Detection (Count: {len(cells)})")
        st.image(res_disp, use_column_width=True)
    
    st.success(f"Successfully designed mask and detected {len(cells)} cells.")
