# 🔬 Cell Counter

A web-based biological image analysis tool built with Python and Streamlit. This application automates the counting of cells in microfluidic channels using Watershed segmentation, White Top-Hat filtering, and dynamic shape analysis.

## Features
- **Upload & Go:** Supports `.jpg`, `.png`, and `.tif` formats.
- **Background Elimination:** Automatically ignores dark channel walls, scratches, and uneven lighting.
- **Real-Time Parameter Tuning:** Adjust brightness sensitivity, cell roundness, and size thresholds on the fly.
- **De-clumping:** Uses Distance Transforms and Watershed algorithms to accurately separate touching cells.

## Installation & Usage

1. **Install dependencies:**
   Open your terminal/command prompt, navigate to this folder, and run:
   ```bash
   pip install -r requirements.txt