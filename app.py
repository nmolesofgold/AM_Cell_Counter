import streamlit as st
import numpy as np
import cv2
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image

st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter (High-Speed Version)")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Controls")
sensitivity = st.sidebar.slider("Sensitivity", 0.1, 1.5, 0.5, 0.1)
roundness = st.sidebar.slider("Roundness Tolerance", 0.5, 1.0, 0.95, 0.05)

st.sidebar.subheader("Boundary Edits")
# This allows you to "edit out" the top/bottom/sides if there are scratches
y_trim = st.sidebar.slider("Vertical Trim (Top/Bottom)", 0, 200, 50)
x_trim = st.sidebar.slider("Horizontal Trim (Left/Right)", 0, 200, 20)

st.sidebar.subheader("Size Filters")
min_area = st.sidebar.number_input("Min Size", 1, 100, 15)
max_area = st.sidebar.number_input("Max Size", 100, 2000, 800)

@st.cache_data
def run_analysis(image_array, sens, round_val, min_a, max_a, yt, xt):
    # 1. Grayscale & Trim (Manual "Editing" of borders)
    if len(image_array.shape) == 3:
        img = color.rgb2gray(image_array[:, :, :3])
    else:
        img = image_array
    
    # Create a copy for the mask
    h, w = img.shape
    mask = np.ones_like(img)
    # Edit out the areas you don't want by setting mask to 0
    mask[:yt, :] = 0 # Top
    mask[-yt:, :] = 0 # Bottom
    mask[:, :xt] = 0 # Left
    mask[:, -xt:] = 0 # Right
    
    # 2. Top-Hat & Threshold
    clean = np.copy(img)
    clean[-50:, :] = 0 # Scale bar wipe
    top_hat = morphology.white_tophat(clean, morphology.disk(15))
    bw = (top_hat > (filters.threshold_otsu(top_hat) * sens)) & (mask > 0)

    # 3. Watershed
    dist = ndi.distance_transform_edt(bw)
    peaks = feature.peak_local_max(dist, min_distance=4, labels=bw)
    m = np.zeros(dist.shape, dtype=bool)
    if len(peaks) > 0: m[tuple(peaks.T)] = True
    markers, _ = ndi.label(m)
    labels = segmentation.watershed(-dist, markers, mask=bw)

    # 4. Filter
    regions = measure.regionprops(labels)
    valid = [r for r in regions if min_a <= r.area <= max_a and r.eccentricity < round_val]
    
    return img, valid

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "tif"])

if uploaded_file:
    raw_img = np.array(Image.open(uploaded_file))
    
    # Run the math
    gray_img, cells = run_analysis(raw_img, sensitivity, roundness, min_area, max_area, y_trim, x_trim)
    
    # --- FAST DRAWING ---
    # Convert grayscale to BGR so we can draw RED circles
    output_img = (gray_img * 255).astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
    
    for c in cells:
        y, x = c.centroid
        radius = int(c.equivalent_diameter_area / 2 * 1.2)
        # Use OpenCV to draw directly on the pixels (lightning fast)
        cv2.circle(output_img, (int(x), int(y)), radius, (255, 0, 0), 2)

    col1, col2 = st.columns(2)
    col1.image(raw_img, caption="Original", use_column_width=True)
    col2.image(output_img, caption=f"Count: {len(cells)}", use_column_width=True)
    
    st.success(f"Detected {len(cells)} cells.")
