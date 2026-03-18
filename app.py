import streamlit as st
import numpy as np
import cv2
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image
import pandas as pd
import io

st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter: High-Res Shape Designer")

# --- SIDEBAR: CHIP SHAPE DESIGNER ---
st.sidebar.header("📐 Chip Shape Designer")
st.sidebar.caption("Define the exact internal geometry. Units are in Pixels (px).")

# High-Range side controls
with st.sidebar.expander("✂️ Boundary Trims (px)", expanded=True):
    t_trim = st.slider("Top Trim (px)", 0, 2000, 100)
    b_trim = st.slider("Bottom Trim (px)", 0, 2000, 100)
    l_trim = st.slider("Left Trim (px)", 0, 2000, 50)
    r_trim = st.slider("Right Trim (px)", 0, 2000, 50)

# High-Range corner controls
with st.sidebar.expander("🌀 Corner Rounding (px)", expanded=True):
    st.caption("Set radius to half the channel height for a perfect semicircle.")
    tl_rad = st.slider("Top-Left Radius (px)", 0, 3000, 100)
    tr_rad = st.slider("Top-Right Radius (px)", 0, 3000, 100)
    bl_rad = st.slider("Bottom-Left Radius (px)", 0, 3000, 100)
    br_rad = st.slider("Bottom-Right Radius (px)", 0, 3000, 100)

st.sidebar.markdown("---")

# Detection Settings
st.sidebar.header("⚙️ Detection Settings")
sensitivity = st.sidebar.slider("Sensitivity (x)", 0.1, 1.5, 0.5, 0.1)
roundness = st.sidebar.slider("Roundness (Eccentricity)", 0.5, 1.0, 0.95, 0.05)

# Size Filters
st.sidebar.subheader("📏 Size Filters (px²)")
min_area = st.sidebar.number_input("Min Area (px²)", 1, 500, 15)
max_area = st.sidebar.number_input("Max Area (px²)", 100, 10000, 800)

# --- MASKING ENGINE ---

def create_custom_mask(h, w, t, b, l, r, tl, tr, bl, br):
    mask = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = t, h - b
    x1, x2 = l, w - r
    
    if y2 <= y1 or x2 <= x1: return mask.astype(bool)

    # 1. Draw the base rectangle
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    
    # 2. Subtract the corners (Inverse Masking)
    # This method is more robust for massive radii
    corner_mask = np.ones((h, w), dtype=np.uint8)
    
    # Top-Left
    if tl > 0:
        cv2.rectangle(corner_mask, (x1, y1), (x1+tl, y1+tl), 0, -1)
        cv2.circle(corner_mask, (x1+tl, y1+tl), tl, 1, -1)
    # Top-Right
    if tr > 0:
        cv2.rectangle(corner_mask, (x2-tr, y1), (x2, y1+tr), 0, -1)
        cv2.circle(corner_mask, (x2-tr, y1+tr), tr, 1, -1)
    # Bottom-Left
    if bl > 0:
        cv2.rectangle(corner_mask, (x1, y2-bl), (x1+bl, y2), 0, -1)
        cv2.circle(corner_mask, (x1+bl, y2-bl), bl, 1, -1)
    # Bottom-Right
    if br > 0:
        cv2.rectangle(corner_mask, (x2-br, y2-br), (x2, y2), 0, -1)
        cv2.circle(corner_mask, (x2-br, y2-br), br, 1, -1)
        
    final_mask = cv2.bitwise_and(mask, corner_mask)
    return final_mask.astype(bool)

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
uploaded_file = st.file_uploader("Upload Microscopy Image", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    raw_img = np.array(Image.open(uploaded_file))
    if len(raw_img.shape) == 3:
        gray_img = color.rgb2gray(raw_img[:, :, :3])
    else:
        gray_img = raw_img
    
    h, w = gray_img.shape
    
    # Update Mask (Instant Preview)
    analysis_mask = create_custom_mask(h, w, t_trim, b_trim, l_trim, r_trim, tl_rad, tr_rad, bl_rad, br_rad)
    
    # Red Overlay for "Edited Out" areas
    overlay = (gray_img * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    red_shroud = np.zeros_like(overlay)
    red_shroud[:] = [255, 0, 0]
    overlay[~analysis_mask] = cv2.addWeighted(overlay[~analysis_mask], 0.7, red_shroud[~analysis_mask], 0.3, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Align Chip Shape")
        st.image(overlay, use_column_width=True)
        run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    with col2:
        st.subheader("2. Analysis Results")
        if run_btn:
            with st.spinner("Processing..."):
                cells = analyze_cells(gray_img, analysis_mask, sensitivity, roundness, min_area, max_area)
                res_disp = (gray_img * 255).astype(np.uint8)
                res_disp = cv2.cvtColor(res_disp, cv2.COLOR_GRAY2RGB)
                
                # Data for CSV export
                data = []
                for c in cells:
                    y, x = c.centroid
                    rad = int(c.equivalent_diameter_area / 2 * 1.2)
                    cv2.circle(res_disp, (int(x), int(y)), rad, (0, 255, 0), 2)
                    data.append({"Cell_ID": len(data)+1, "X": x, "Y": y, "Area_px2": c.area})
                
                st.image(res_disp, use_column_width=True)
                st.metric("Final Cell Count", len(cells))
                
                # --- EXPORT SECTION ---
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"count_result_{len(cells)}.csv",
                    mime='text/csv',
                )
        else:
            st.info("Align the red shroud with your chip walls, then click 'Run Analysis'.")
