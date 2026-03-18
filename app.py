import streamlit as st
import numpy as np
import cv2
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image

st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter: Precision Masking")

# --- SIDEBAR: CHIP SHAPE DESIGNER ---
st.sidebar.header("📐 Chip Shape Designer")
st.sidebar.caption("Align the bright area with the inside of your chip.")

# 4-Side Independent Trimming
t_trim = st.sidebar.slider("Top Trim", 0, 800, 100)
b_trim = st.sidebar.slider("Bottom Trim", 0, 800, 100)
l_trim = st.sidebar.slider("Left Trim", 0, 800, 50)
r_trim = st.sidebar.slider("Right Trim", 0, 800, 50)

corner_radius = st.sidebar.slider("Corner Rounding (Semicircle)", 0, 400, 100)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Detection Settings")
sensitivity = st.sidebar.slider("Sensitivity (Brightness)", 0.1, 1.5, 0.5, 0.1, help="Lower = Catch faint cells")
roundness = st.sidebar.slider("Roundness Tolerance", 0.5, 1.0, 0.95, 0.05, help="1.0 = Allow ovals")

st.sidebar.subheader("📏 Size Filters (Pixels)")
min_area = st.sidebar.number_input("Min Cell Size", 1, 100, 15, help="Filters out tiny dust/specks")
max_area = st.sidebar.number_input("Max Cell Size", 100, 5000, 800, help="Filters out large scratches/clumps")

# --- FUNCTIONS ---

def create_rounded_mask(h, w, t, b, l, r, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = t, h - b
    x1, x2 = l, w - r
    
    # Safety check for radius
    radius = max(0, min(radius, (y2 - y1) // 2, (x2 - x1) // 2))
    
    # Draw the rounded rectangle
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 1, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 1, -1)
    if radius > 0:
        cv2.circle(mask, (x1 + radius, y1 + radius), radius, 1, -1)
        cv2.circle(mask, (x2 - radius, y1 + radius), radius, 1, -1)
        cv2.circle(mask, (x1 + radius, y2 - radius), radius, 1, -1)
        cv2.circle(mask, (x2 - radius, y2 - radius), radius, 1, -1)
    return mask.astype(bool)

@st.cache_data
def analyze_cells(img, mask, sens, round_val, min_a, max_a):
    """ Heavy math triggered only by the button """
    # Scale bar wipe
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
    # 1. Load Image
    raw_img = np.array(Image.open(uploaded_file))
    if len(raw_img.shape) == 3:
        gray_img = color.rgb2gray(raw_img[:, :, :3])
    else:
        gray_img = raw_img
    
    h, w = gray_img.shape
    
    # 2. Update Mask (Instant)
    analysis_mask = create_rounded_mask(h, w, t_trim, b_trim, l_trim, r_trim, corner_radius)
    
    # 3. Create Shaded Overlay Preview
    # Create a red-tinted version of the image
    overlay = (gray_img * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    
    # Create a red 'shroud' for ignored areas
    red_shroud = np.zeros_like(overlay)
    red_shroud[:] = [255, 0, 0] # Red color
    
    # Blend the original with red for ignored areas (0.3 alpha)
    ignored_area = ~analysis_mask
    overlay[ignored_area] = cv2.addWeighted(overlay[ignored_area], 0.7, red_shroud[ignored_area], 0.3, 0)
    
    # --- DISPLAY LAYOUT ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: Design Mask")
        st.caption("Red area is IGNORED. Align the bright center with your channel.")
        st.image(overlay, use_column_width=True)
        
        # THE BIG BUTTON: Prevents the app from running heavy math on every slider move
        run_btn = st.button("🚀 Start Cell Count", use_container_width=True)

    with col2:
        st.subheader("Step 2: Analysis Results")
        
        # Only run if the button is clicked or if we've already run it once
        if run_btn:
            with st.spinner("Processing Watershed Analysis..."):
                cells = analyze_cells(gray_img, analysis_mask, sensitivity, roundness, min_area, max_area)
                
                # Draw circles on results
                res_disp = (gray_img * 255).astype(np.uint8)
                res_disp = cv2.cvtColor(res_disp, cv2.COLOR_GRAY2RGB)
                for c in cells:
                    y, x = c.centroid
                    rad = int(c.equivalent_diameter_area / 2 * 1.2)
                    cv2.circle(res_disp, (int(x), int(y)), rad, (0, 255, 0), 2) # Green circles for clarity
                
                st.image(res_disp, use_column_width=True)
                st.metric("Total Count", len(cells))
                st.success("Analysis Complete!")
        else:
            st.info("Adjust the mask on the left, then click 'Start Cell Count'.")
