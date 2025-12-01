import os
import io
from typing import List, Optional

import requests
import streamlit as st
from PIL import Image
import replicate


# ------------------------------
# Config – Replicate model + token
# ------------------------------
# Public Stable Diffusion img2img model on Replicate
REPLICATE_MODEL_ID = "stability-ai/stable-diffusion-img2img"


def get_replicate_token() -> Optional[str]:
    """
    Resolve the Replicate API token.
    Priority:
    1. Streamlit secrets (REPLICATE_API_TOKEN)
    2. Environment variable REPLICATE_API_TOKEN
    """
    token = None

    # Try Streamlit secrets (Streamlit Cloud: Settings -> Secrets)
    try:
        token = st.secrets.get("REPLICATE_API_TOKEN", None)
    except Exception:
        token = None

    # Fallback to environment variable
    if not token:
        token = os.getenv("REPLICATE_API_TOKEN")

    return token


# ------------------------------
# Utility functions
# ------------------------------
def load_image(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert("RGB")
    return img


def resize_to_512(img: Image.Image) -> Image.Image:
    # Simple center-ish resize; for PoC this is fine
    return img.resize((512, 512))


def pil_to_bytes_io(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ------------------------------
# Replicate diffusion call
# ------------------------------
def generate_thermo_variants(
    base_image: Image.Image,
    prompt: str,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    steps: int = 20,
    num_outputs: int = 3,
) -> List[Image.Image]:
    """
    Call Replicate's Stable Diffusion img2img model with Image B (thermography)
    and return a list of PIL Images (synthetic variants).
    """
    token = get_replicate_token()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. "
            "Please configure it in Streamlit Secrets or as an environment variable."
        )

    # Replicate Python client reads this env var
    os.environ["REPLICATE_API_TOKEN"] = token

    # Prepare image buffer (file-like object) for Replicate
    base_image = resize_to_512(base_image)
    img_buf = pil_to_bytes_io(base_image)

    # Run model on Replicate
    # NOTE: Different models can have slightly different inputs; these match the usual SD img2img schema.
    output = replicate.run(
        REPLICATE_MODEL_ID,
        input={
            "image": img_buf,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "num_outputs": num_outputs,
        },
    )

    # Replicate typically returns a list of URLs (strings)
    images: List[Image.Image] = []
    for item in output:
        if isinstance(item, str):
            # Assume URL
            resp = requests.get(item)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            images.append(img)
        else:
            # Fallback: if the object is already bytes-like (defensive)
            try:
                data = bytes(item)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                images.append(img)
            except Exception:
                # Skip if can't decode
                continue

    if not images:
        raise RuntimeError("No images returned from Replicate.")

    return images


# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.set_page_config(
        page_title="Thermography Diffusion (Replicate)",
        layout="wide",
    )

    st.title("🩺 Thermography Augmentation via Diffusion (Replicate API)")
    st.caption(
        "Upload Image A (mammogram, reference) and Image B (thermography, base). "
        "This app uses Replicate's Stable Diffusion img2img model to generate "
        "three synthetic variants of Image B.\n\n"
        "⚠️ Research/demo only – not for clinical diagnosis."
    )

    token_present = bool(get_replicate_token())
    with st.expander("Replicate configuration", expanded=False):
        if token_present:
            st.success("REPLICATE_API_TOKEN detected ✅")
        else:
            st.warning(
                "REPLICATE_API_TOKEN is not set. Go to Streamlit Cloud → Settings → Secrets and add:\n\n"
                "REPLICATE_API_TOKEN = \"r8_...\""
            )

    # Layout for uploads
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Image A: Mammogram (reference)")
        file_a = st.file_uploader(
            "Upload Image A (optional)", type=["png", "jpg", "jpeg"], key="image_a"
        )
        img_a = load_image(file_a)
        if img_a is not None:
            st.image(img_a, caption="Image A (Mammogram)", use_column_width=True)
        else:
            st.info("Image A is optional and used only for display/context.")

    with col_b:
        st.subheader("Image B: Thermography (base)")
        file_b = st.file_uploader(
            "Upload Image B (required)", type=["png", "jpg", "jpeg"], key="image_b"
        )
        img_b = load_image(file_b)
        if img_b is not None:
            st.image(img_b, caption="Image B (Thermography Base)", use_column_width=True)
        else:
            st.warning("Please upload Image B. This is the image used for diffusion.")

    st.markdown("---")

    # Generation controls
    st.subheader("Generation settings")

    prompt = st.text_area(
        "Prompt guiding the synthetic thermography:",
        value=(
            "medical breast thermography heatmap, realistic, high quality, "
            "consistent anatomy, clean clinical style"
        ),
        height=80,
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        strength = st.slider(
            "Transformation strength",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="0.1 = very subtle change, 0.9 = strong transformation from the base image.",
        )
    with col_s2:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=3.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
            help="Higher values make the output follow the text prompt more strongly.",
        )
    with col_s3:
        steps = st.slider(
            "Diffusion steps",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="More steps can improve quality but are slower and cost more on Replicate.",
        )

    st.markdown("---")

    generate_btn = st.button("🚀 Generate 3 synthetic thermography images")

    if generate_btn:
        if img_b is None:
            st.error("Please upload Image B (thermography) before generating.")
            return

        if not token_present:
            st.error(
                "REPLICATE_API_TOKEN is not configured. "
                "Please set it in Streamlit Secrets or as an environment variable."
            )
            return

        with st.spinner("Calling Replicate and generating images…"):
            try:
                images = generate_thermo_variants(
                    base_image=img_b,
                    prompt=prompt,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    steps=steps,
                    num_outputs=3,
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

        st.success("Done! Here are the 3 synthetic variants of Image B:")
        cols = st.columns(3)
        for col, img in zip(cols, images):
            with col:
                st.image(img, use_column_width=True)


if __name__ == "__main__":
    main()
