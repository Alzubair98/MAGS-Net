"""Streamlit research interface. Run: python -m streamlit run app.py"""
import hashlib
import logging
import sys
from pathlib import Path

# Support both `streamlit run app.py` and `streamlit run magsnet_demo/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import torch

from magsnet_demo.config import (APP_DIR, APP_NAME, CLASS_NAMES, DEVICE, DISCLAIMER,
                                IMAGE_SIZE, INPUT_LIMITATIONS, MANIFEST, SEGMENTATION_THRESHOLD)
from magsnet_demo.inference import load_ensemble, predict, resource_key
from magsnet_demo.preprocessing import prepare_image
from magsnet_demo.utils import DemoError
from magsnet_demo.visualization import make_downloads, render_outputs

LOGGER = logging.getLogger("magsnet_demo")


@st.cache_resource(show_spinner=False)
def cached_ensemble(checkpoint_key, device_name):
    return load_ensemble(device_name)


def clear_image():
    for key in ("image_digest", "prepared", "result", "visuals", "downloads", "result_device"):
        st.session_state.pop(key, None)
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def report_error(exc):
    # Do not record filenames, bytes, arrays, or patient metadata.
    LOGGER.error("MAGS-Net operation failed (%s)", type(exc).__name__)
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        st.error("CUDA ran out of memory. Select CPU in the sidebar and analyze the image again.")
    elif isinstance(exc, DemoError):
        st.error(str(exc))
    else:
        st.error("Analysis could not be completed. Check the image, model files and dependencies, then try again.")


def show_results(prepared, result, visuals, downloads):
    st.markdown("### Analysis results")
    a, b, c = st.columns([1.15, 1, 1.15])
    a.metric("Predicted tumor type", CLASS_NAMES[result["predicted_label"]], border=True)
    b.metric("Model probability", f"{result['probabilities'].max():.1%}", border=True)
    c.metric("Predicted lesion area fraction", f"{result['lesion_area_fraction']:.2%}", border=True)
    st.caption("Model probabilities are not clinical confidence. A displayed tumor label or mask "
               "does not establish that a tumor is present; the model has no tumor-free class.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Original MRI**")
        st.image(prepared.original_rgb, width="stretch")
    with right:
        st.markdown("**Segmentation overlay**")
        st.image(visuals["overlay"], width="stretch")
    st.caption("Teal indicates the predicted lesion. The binary mask is mapped back to the original image "
               "using the detected crop coordinates and nearest-neighbor interpolation.")

    with st.container(border=True):
        st.markdown("#### Class probabilities")
        for name, probability in zip(CLASS_NAMES, result["probabilities"]):
            st.progress(float(probability), text=f"{name}  ·  {probability:.2%}")
        st.caption("Equal-weight mean of softmax probabilities from the five outer-fold models.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Predicted mask**")
        st.image(visuals["mask"], width="stretch", clamp=True)
    with right:
        st.markdown("**Lesion probability map**")
        st.image(visuals["probability_map"], width="stretch")
        st.caption("Within the crop: purple ≈ 0 → yellow ≈ 1. Black outside the crop: not analyzed.")
    st.caption("Area fraction = predicted tumor pixels / 50,176 pixels in the analyzed 224×224 crop. "
               "It describes a 2D image, not physical tumor volume. Areas outside the crop are not assessed.")

    with st.expander("Model Insight: Adaptive Gates"):
        st.markdown("#### Adaptive Multi-Scale Gate Response")
        data = {"Stage": [f"Stage {i + 1}" for i in range(3) for _ in range(2)],
                "Mean gate weight": result["gates"].reshape(-1).tolist(),
                "Scale": ["3×3", "5×5"] * 3}
        st.bar_chart(data, x="Stage", y="Mean gate weight", color="Scale",
                     stack="normalize", horizontal=True)
        st.table([{"Stage": f"Stage {i + 1}", "α3": f"{g[0]:.4f}", "α5": f"{g[1]:.4f}"}
                  for i, g in enumerate(result["gates"])])
        st.caption("Each stage averages channel weights within each block, then its blocks, then the five models. "
                   "The two weights sum to one. These responses are not calibrated explanations of causality.")

    with st.expander("Preprocessing & analyzed field of view"):
        x1, y1, x2, y2 = prepared.crop_box
        st.image(prepared.analyzed_uint8, caption="Exact image supplied to the models", width=224)
        st.write(f"Original: {prepared.original_gray.shape[1]}×{prepared.original_gray.shape[0]} pixels. "
                 f"Crop: x=[{x1}, {x2}), y=[{y1}, {y2}). Input: 1×224×224.")
        st.write("OpenCV grayscale → per-image min–max uint8 → MRI-derived contour crop → "
                 "linear resize → float32 / 255. No ground-truth mask, augmentation or ImageNet normalization.")

    st.markdown("#### Download results")
    columns = st.columns(3)
    items = [("Binary mask", "mask", "magsnet_mask.png", "image/png"),
             ("Segmentation overlay", "overlay", "magsnet_overlay.png", "image/png"),
             ("Analysis summary", "summary", "magsnet_summary.json", "application/json")]
    for column, (label, key, filename, mime) in zip(columns, items):
        column.download_button(label, downloads[key], file_name=filename, mime=mime,
                               on_click="ignore", width="stretch")
    st.caption("Downloads are created in memory and contain no uploaded filename. The JSON records the analyzed grid, "
               "crop coordinates, class probabilities, gates and checkpoint fingerprints.")


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
    st.markdown(f"<style>{(APP_DIR / 'assets/style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.markdown('<div class="research-badge">RESEARCH PROTOTYPE · UNDER DEVELOPMENT</div>', unsafe_allow_html=True)
    st.title("MAGS-Net")
    st.markdown('<div class="subtitle">Lightweight Adaptive Multi-Scale Gated Network<br>'
                'for Joint Brain Tumor Classification and Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)
    st.warning("**Research use only — under development**\n\n" + DISCLAIMER
               + "\n\n**Current input limitations:** " + INPUT_LIMITATIONS)

    with st.sidebar:
        st.markdown("### ABOUT MAGS-Net")
        st.write("Joint Classification + Segmentation")
        st.caption("3 tumor classes · Lightweight architecture")
        st.metric("Parameters / model", f"{MANIFEST['parameters_per_model']:,}")
        st.write(f"**{MANIFEST['conv_linear_macs_per_model'] / 1e9:.3f} G MACs** / model")
        st.write(f"**{MANIFEST['fp32_weights_mib_per_model']:.3f} MiB** FP32 weights / model")
        st.caption("MACs count Conv2d/Linear operations. Five-model inference has approximately "
                   f"{5 * MANIFEST['conv_linear_macs_per_model'] / 1e9:.3f} G MACs and "
                   f"{5 * MANIFEST['fp32_weights_mib_per_model']:.3f} MiB weights, excluding runtime buffers.")
        st.divider()
        st.markdown("### MODEL STATUS")
        options = ["CUDA", "CPU"] if DEVICE == "cuda" else ["CPU"]
        choice = st.selectbox("Inference device", options, key="device_choice")
        device = choice.lower()
        status = st.empty()
        st.write(f"Image size: **{IMAGE_SIZE}×{IMAGE_SIZE} grayscale**")
        st.write(f"Segmentation threshold: **≥ {SEGMENTATION_THRESHOLD}**")
        st.caption("Seed 42 · frozen FP32 weights")
        st.button("Clear uploaded image & results", on_click=clear_image, width="stretch")
        st.divider()
        st.markdown("**Research use only · Under development.**\n\nNot intended for clinical diagnosis.")
        st.caption("Images and results stay in this session's memory. No external inference service is used.")

    with st.container(border=True):
        st.markdown("### Upload Brain MRI")
        st.write("Upload a contrast-enhanced T1-weighted brain MRI image to obtain tumor-type probabilities "
                 "and a predicted lesion segmentation.")
        st.caption("Use only de-identified research brain MRI images with a known tumor in one of the "
                   "three supported classes. Tumor-free and non-brain images are outside the model's intended scope.")
        uploaded = st.file_uploader("PNG · JPG · JPEG", type=["png", "jpg", "jpeg"],
                                    key=f"upload_{st.session_state.get('upload_version', 0)}",
                                    help="One 8-bit image; maximum 10 MiB. Upload the original exported image, not an already cropped model input.")
        st.caption("Image only · no ground-truth mask required · processed in memory")

    if uploaded is None:
        for key in ("image_digest", "prepared", "result", "visuals", "downloads", "result_device"):
            st.session_state.pop(key, None)
        status.caption("Models: ready to load 5-fold ensemble" if not st.session_state.get("loaded_device")
                       else f"Models: 5-fold ensemble loaded on {st.session_state['loaded_device'].upper()}")
        st.info("Three-class research model: Meningioma · Glioma · Pituitary. "
                "A tumor-free class and scan-quality validation are not included.")
        st.caption(DISCLAIMER)
        return

    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != st.session_state.get("image_digest") or "prepared" not in st.session_state:
        for key in ("image_digest", "prepared", "result", "visuals", "downloads", "result_device"):
            st.session_state.pop(key, None)
        try:
            st.session_state["prepared"] = prepare_image(data)
            st.session_state["image_digest"] = digest
        except Exception as exc:
            report_error(exc)
            status.caption("Models: inference not started")
            st.caption(DISCLAIMER)
            return
    if st.session_state.get("result_device") not in (None, device):
        for key in ("result", "visuals", "downloads", "result_device"):
            st.session_state.pop(key, None)

    prepared = st.session_state["prepared"]
    preview = st.empty()
    if "result" not in st.session_state:
        with preview.container():
            left, right = st.columns([1, 1.2], gap="large")
            with left:
                st.image(prepared.original_rgb, caption="Original MRI", width=360)
            with right:
                st.markdown("#### Ready for analysis")
                st.write("The five frozen outer-fold models will analyze the same brain crop. "
                         "Classification and lesion probabilities are averaged across all five models.")
                st.caption("No training, fine-tuning, ground-truth masks or test-based model selection.")
    if st.button("Analyze MRI", type="primary", width="stretch"):
        for key in ("result", "visuals", "downloads", "result_device"):
            st.session_state.pop(key, None)
        try:
            with st.spinner("Analyzing MRI with MAGS-Net..."):
                ensemble = cached_ensemble(resource_key(), device)
                st.session_state["loaded_device"] = device
                result = predict(ensemble, prepared.tensor)
                visuals = render_outputs(prepared, result)
                downloads = make_downloads(prepared, result, visuals, device)
                st.session_state.update(result=result, visuals=visuals, downloads=downloads, result_device=device)
                preview.empty()
        except Exception as exc:
            report_error(exc)

    loaded = st.session_state.get("loaded_device")
    if loaded:
        status.success(f"5-fold ensemble loaded · {loaded.upper()}")
    else:
        status.caption("Models: ready to load 5-fold ensemble")
    if "result" in st.session_state:
        show_results(prepared, st.session_state["result"], st.session_state["visuals"], st.session_state["downloads"])
    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
