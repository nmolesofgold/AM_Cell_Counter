import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cell Counter", layout="wide")
st.title("🔬 Cell Counter")
st.markdown("Upload a microfluidic microscopy image to automatically count cells using Watershed segmentation and White Top-Hat filtering.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Analysis Parameters")

sensitivity = st.sidebar.slider(
    "Sensitivity (Brightness)", 
    min_value=0.1, max_value=1.5, value=0.5, step=0.1,
    help="Lower = Catches faint, dim cells. Higher = Only strict, glowing cells."
)

roundness = st.sidebar.slider(
    "Roundness (Eccentricity)", 
    min_value=0.5, max_value=1.0, value=0.95, step=0.05,
    help="1.0 = Allows ovals & dividing cells. 0.5 = Strictly perfect circles only."
)

st.sidebar.markdown("### Size Filters")
min_area = st.sidebar.number_input("Minimum Cell Size (px)", min_value=1, value=5, step=1)
max_area = st.sidebar.number_input("Maximum Cell Size (px)", min_value=10, value=800, step=10)

# --- CACHED PROCESSING FUNCTION ---
# st.cache_data prevents the app from re-running the heavy math unless a slider changes
@st.cache_data
def process_image(image_array, sens, round_val, min_a, max_a):
    # 1. Grayscale Conversion
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        gray_image = color.rgb2gray(image_array)
    else:
        gray_image = image_array

    # 2. Scale Bar Wipe (Bottom 50 pixels)
    gray_clean = np.copy(gray_image)
    gray_clean[-50:, :] = 0

    # 3. White Top-Hat
    top_hat = morphology.white_tophat(gray_clean, morphology.disk(15))
    
    # 4. Thresholding
    base_thresh = filters.threshold_otsu(top_hat)
    bw = top_hat > (base_thresh * sens)

    # 5. Peak Detection & Watershed
    distance = ndi.distance_transform_edt(bw)
    coords = feature.peak_local_max(distance, min_distance=4, labels=bw)
    mask = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0: 
        mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=bw)

    # 6. Filtering
    all_regions = measure.regionprops(labels)
    valid_regions = [r for r in all_regions if min_a <= r.area <= max_a and r.eccentricity < round_val]
    
    return gray_image, valid_regions

# --- MAIN APP UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Read the image using PIL and convert to numpy array
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Run processing with a loading spinner
    with st.spinner('Analyzing image...'):
        gray_image, valid_regions = process_image(image_array, sensitivity, roundness, min_area, max_area)

    cell_count = len(valid_regions)

    # Layout for side-by-side images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.imshow(gray_image, cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        st.subheader(f"Detected Cells: {cell_count}")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.imshow(gray_image, cmap='gray')
        ax2.axis('off')

        for props in valid_regions:
            y, x = props.centroid
            r = props.equivalent_diameter_area / 2.0
            c = plt.Circle((x, y), r * 1.2, color='red', linewidth=1, fill=False)
            ax2.add_patch(c)
            
        st.pyplot(fig2)

    st.success(f"Successfully detected {cell_count} cells!")
else:
    st.info("Please upload an image to begin.")