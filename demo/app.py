import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
from hashlib import md5

# Local imports from project
import sys, os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.compress import compress_image
from src.metrics import psnr_hvs, ssim_index

st.set_page_config(page_title="Adaptive k-Means Palette Compressor", layout="centered")
st.title("Adaptive k-Means Color-Palette Compression demo")

st.markdown("Upload any RGB image (PNG/JPEG ≤ 4 MP).  The app compresses it to a palette-optimized PNG-8 and reports basic quality metrics.")

uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

# Session-state keys: last_hash, comp_bytes, orig_img, comp_img, metrics

if uploaded is not None:
    # Compute a hash of the uploaded file to avoid recomputing on each rerun
    file_hash = md5(uploaded.getbuffer()).hexdigest()

    if st.session_state.get("last_hash") != file_hash:
        with st.spinner("Running compression …"):
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_in.write(uploaded.getbuffer())
            tmp_in.flush()
            tmp_out_path = Path(tempfile.mkstemp(suffix=".png")[1])

            compress_image(Path(tmp_in.name), tmp_out_path)

            orig_img = Image.open(tmp_in.name).convert("RGB")
            comp_img = Image.open(tmp_out_path).convert("RGB")
            orig_arr = np.asarray(orig_img).astype(np.float32) / 255.0
            comp_arr = np.asarray(comp_img).astype(np.float32) / 255.0

            psnr_value = psnr_hvs(comp_arr, orig_arr)
            ssim_value = ssim_index(comp_arr, orig_arr)
            comp_size = tmp_out_path.stat().st_size
            ratio = comp_size / uploaded.size

            # store in session_state
            st.session_state.update({
                "last_hash": file_hash,
                "orig_img": orig_img,
                "comp_img": comp_img,
                "comp_bytes": tmp_out_path.read_bytes(),
                "psnr": psnr_value,
                "ssim": ssim_value,
                "ratio": ratio,
                "comp_size": comp_size
            })

            os.unlink(tmp_in.name)
            os.unlink(tmp_out_path)
    # retrieve from state
    orig_img = st.session_state["orig_img"]
    comp_img = st.session_state["comp_img"]
    psnr_value = st.session_state["psnr"]
    ssim_value = st.session_state["ssim"]
    ratio = st.session_state["ratio"]
    comp_size = st.session_state["comp_size"]

    col1, col2 = st.columns(2)
    # VERSION-COMPATIBLE image display helper
    def _show_image(img):
        try:
            # Newer Streamlit (≥1.40) prefers use_container_width
            st.image(img, use_container_width=True)
        except TypeError:
            # Older Streamlit (<1.40) still uses use_column_width
            st.image(img, use_column_width=True)

    with col1:
        st.subheader("Original")
        _show_image(orig_img)
        st.text(f"{uploaded.size/1024:.1f} kB")
    with col2:
        st.subheader("Compressed PNG-8")
        _show_image(comp_img)
        st.text(f"{comp_size/1024:.1f} kB  ({ratio*100:.1f}% of original)")

    st.markdown("### Quality metrics")
    st.write({"PSNR-HVS (dB)": round(psnr_value, 2), "SSIM": round(ssim_value, 4)})

    # Download button
    st.download_button("Download compressed image", st.session_state["comp_bytes"], file_name="compressed.png", mime="image/png") 