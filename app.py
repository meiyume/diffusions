import os
import io
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
import replicate
import numpy as np
import cv2

# ------------------------------
# Replicate model IDs & token
# ------------------------------

# Your existing vanilla SD img2img model (uncontrolled baseline)
SD_IMG2IMG_MODEL_ID = (
    "stability-ai/stable-diffusion-img2img:"
    "15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d"
)

# ControlNet canny model (structure-controlled run)
# You can pin a specific version later if you like; this uses latest.
CONTROLNET_MODEL_ID = "jagilley/controlnet-canny"


def get_replicate_token() -> Optional[str]:
    """Resolve Replicate API token from Streamlit secrets or env."""
    token = None
    try:
        token = st.secrets.get("REPLICATE_API_TOKEN", None)
    except Exception:
        token = None

    if not token:
        token = os.getenv("REPLICATE_API_TOKEN")

    return token


# ------------------------------
# Image utilities
# ------------------------------
def load_image(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert("RGB")
    return img


def resize_to_512(img: Image.Image) -> Image.Image:
    return img.resize((512, 512))


def pil_to_bytes_io(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def compute_structural_priors(
    img: Image.Image,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    From thermography image B, compute:
      - Canny edges
      - Silhouette mask
      - Hybrid (edges within silhouette)

    Returned as 512x512 single-channel PIL images.
    """
    img_resized = resize_to_512(img)
    img_np = np.array(img_resized)

    # grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # ---- Canny edges ----
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ---- Silhouette via Otsu + largest contour ----
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    silhouette = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(silhouette, [largest], -1, 255, thickness=-1)

    # ---- Hybrid: edges restricted to silhouette ----
    hybrid = cv2.bitwise_and(edges, edges, mask=silhouette)

    canny_pil = Image.fromarray(edges)
    silhouette_pil = Image.fromarray(silhouette)
    hybrid_pil = Image.fromarray(hybrid)

    return canny_pil, silhouette_pil, hybrid_pil


# ------------------------------
# Replicate helpers
# ------------------------------
def ensure_replicate_token() -> str:
    token = get_replicate_token()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. "
            "Add it in Streamlit Secrets or as an environment variable."
        )
    os.environ["REPLICATE_API_TOKEN"] = token
    return token


def call_sd_img2img(
    base_image: Image.Image,
    prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    num_outputs: int = 3,
) -> List[Image.Image]:
    """Vanilla SD img2img (uncontrolled baseline)."""
    ensure_replicate_token()

    base_image = resize_to_512(base_image)
    img_buf = pil_to_bytes_io(base_image)

    response = replicate.run(
        SD_IMG2IMG_MODEL_ID,
        input={
            "image": img_buf,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "num_outputs": num_outputs,
        },
    )

    if isinstance(response, dict):
        urls = response.get("output", [])
    else:
        urls = response

    if not urls:
        raise RuntimeError("SD img2img returned no output URLs.")

    images: List[Image.Image] = []
    for url in urls:
        try:
            r = requests.get(str(url))
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            images.append(img)
        except Exception as e:
            st.warning(f"Failed to download image from {url}: {e}")

    if not images:
        raise RuntimeError("Failed to download any SD img2img outputs.")

    return images


def call_controlnet_canny(
    control_image: Image.Image,
    prompt: str,
    guidance_scale: float,
    steps: int,
    num_outputs: int = 3,
) -> List[Image.Image]:
    """
    ControlNet Canny run using the cazwaz/controlnet-canny model on Replicate.

    We treat the selected structural prior (Canny / Silhouette / Hybrid)
    as the control image. This model expects:
      - image (canny-like or structural map)
      - prompt
      - num_outputs
      - guidance_scale
      - image_resolution
      - low_threshold, high_threshold

    NOTE: 'steps' is not exposed in this particular model's schema,
    so we ignore it here to avoid 400/422 errors.
    """
    ensure_replicate_token()

    # Prepare control image as PNG buffer
    control_image = resize_to_512(control_image)
    img_buf = pil_to_bytes_io(control_image)

    # Call ControlNet model
    response = replicate.run(
        CONTROLNET_MODEL_ID,
        input={
            "image": img_buf,
            "prompt": prompt,
            "num_outputs": num_outputs,
            "guidance_scale": guidance_scale,
            "image_resolution": 512,
            "low_threshold": 100,
            "high_threshold": 200,
        },
    )

    # Normalise response → list of URL strings
    if isinstance(response, dict):
        urls = response.get("output", [])
    else:
        urls = response

    if not urls:
        raise RuntimeError("ControlNet returned no output URLs.")

    images: List[Image.Image] = []
    for url in urls:
        try:
            r = requests.get(str(url))
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            images.append(img)
        except Exception as e:
            st.warning(f"Failed to download ControlNet image from {url}: {e}")

    if not images:
        raise RuntimeError("Failed to download any ControlNet outputs.")

    return images


# ------------------------------
# Streamlit app
# ------------------------------
def main():
    st.set_page_config(
        page_title="Thermography · SD img2img + ControlNet",
        layout="wide",
    )

    st.title("🩺 Thermography Diffusion: Uncontrolled vs ControlNet-Guided")
    st.caption(
        "Upload mammogram (A) and thermography (B). "
        "First we run vanilla Stable Diffusion img2img on B (uncontrolled). "
        "Then we derive structural priors (Canny / Silhouette / Hybrid) from B, "
        "let you pick one as a ControlNet conditioning image, and run a second, "
        "structure-guided generation.\n\n"
        "⚠️ Research demo only – not for clinical use."
    )

    # Session storage
    if "uncontrolled_images" not in st.session_state:
        st.session_state["uncontrolled_images"] = None
    if "controlled_images" not in st.session_state:
        st.session_state["controlled_images"] = None
    if "priors" not in st.session_state:
        st.session_state["priors"] = None  # (canny, silhouette, hybrid)

    token_present = bool(get_replicate_token())
    with st.expander("Replicate configuration", expanded=False):
        if token_present:
            st.success("REPLICATE_API_TOKEN detected ✅")
        else:
            st.warning(
                "REPLICATE_API_TOKEN is not set. "
                "On Streamlit Cloud: Settings → Secrets → add\n"
                "REPLICATE_API_TOKEN = \"r8_...\""
            )

    # ------------------ Uploads ------------------
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Image A: Mammogram (optional)")
        file_a = st.file_uploader(
            "Upload Image A", type=["png", "jpg", "jpeg"], key="image_a"
        )
        img_a = load_image(file_a)
        if img_a is not None:
            st.image(img_a, caption="Image A (Mammogram)", width=256)
        else:
            st.info("Image A is optional and only for context.")

    with col_b:
        st.subheader("Image B: Thermography (required)")
        file_b = st.file_uploader(
            "Upload Image B", type=["png", "jpg", "jpeg"], key="image_b"
        )
        img_b = load_image(file_b)
        if img_b is not None:
            st.image(img_b, caption="Image B (Thermography Base)", width=256)
        else:
            st.error("Please upload Image B – it is required for all generation.")

    st.markdown("---")

    # ------------------ Shared diffusion settings ------------------
    st.subheader("Diffusion settings (used for both runs)")

    prompt = st.text_area(
        "Prompt:",
        value=(
            "medical breast thermography heatmap, realistic, high quality, "
            "consistent anatomy, clean clinical style"
        ),
        height=80,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        strength = st.slider(
            "img2img strength (for SD img2img baseline)",
            0.1,
            0.9,
            0.6,
            0.05,
            help="Only affects the *uncontrolled* SD img2img run.",
        )
    with c2:
        guidance_scale = st.slider(
            "Guidance / scale",
            3.0,
            15.0,
            7.5,
            0.5,
            help="Higher = follow text more closely.",
        )
    with c3:
        steps = st.slider(
            "Steps (SD + ControlNet)",
            10,
            50,
            20,
            5,
            help="More steps = potentially better, but slower and more cost.",
        )

    st.markdown("---")

    # ------------------ 1) Uncontrolled SD img2img ------------------
    st.subheader("① Uncontrolled diffusion (Image B → SD img2img)")

    if st.button("🚀 Generate 3 uncontrolled images"):
        if img_b is None:
            st.error("Upload Image B first.")
        elif not token_present:
            st.error("REPLICATE_API_TOKEN is not configured.")
        else:
            with st.spinner("Running vanilla SD img2img on Image B…"):
                try:
                    imgs = call_sd_img2img(
                        base_image=img_b,
                        prompt=prompt,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        steps=steps,
                        num_outputs=3,
                    )
                    st.session_state["uncontrolled_images"] = imgs
                except Exception as e:
                    st.error(f"Uncontrolled generation failed: {e}")

    if st.session_state["uncontrolled_images"]:
        cols = st.columns(3)
        for col, img in zip(cols, st.session_state["uncontrolled_images"]):
            with col:
                st.image(img, width=256)
    else:
        st.info("No uncontrolled images yet. Click the button above to generate them.")

    # ------------------ 2) Structural priors from Image B ------------------
    st.markdown("---")
    st.subheader("② Structural priors from Image B (for ControlNet conditioning)")

    if img_b is not None:
        if st.session_state["priors"] is None:
            try:
                canny_img, silhouette_img, hybrid_img = compute_structural_priors(img_b)
                st.session_state["priors"] = (canny_img, silhouette_img, hybrid_img)
            except Exception as e:
                st.error(f"Failed to compute structural priors: {e}")
                st.session_state["priors"] = None

        priors = st.session_state["priors"]
        if priors is not None:
            canny_img, silhouette_img, hybrid_img = priors

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.image(canny_img, caption="Canny edge map", clamp=True, width=256)
            with pc2:
                st.image(
                    silhouette_img,
                    caption="Silhouette mask",
                    clamp=True,
                    width=256,
                )
            with pc3:
                st.image(
                    hybrid_img,
                    caption="Hybrid (edges × silhouette)",
                    clamp=True,
                    width=256,
                )

            st.caption(
                "These are 512×512 structural priors derived from Image B. "
                "ControlNet-canny is trained on canny edges, but you can also try "
                "silhouette or hybrid as conditioning inputs."
            )

            # ------------------ 3) Choose control image & run ControlNet ------------------
            st.subheader("③ Controlled diffusion via ControlNet-Canny")

            choice = st.radio(
                "Choose which structural prior to use as the ControlNet conditioning image:",
                [
                    "Canny edge map (recommended)",
                    "Silhouette mask",
                    "Hybrid (edges × silhouette)",
                ],
                index=0,
            )

            if choice.startswith("Canny"):
                control_base = canny_img
            elif choice.startswith("Silhouette"):
                control_base = silhouette_img
            else:
                control_base = hybrid_img

            if st.button("🎯 Generate 3 controlled images (ControlNet)"):
                if not token_present:
                    st.error("REPLICATE_API_TOKEN is not configured.")
                else:
                    with st.spinner("Running ControlNet-Canny with selected control image…"):
                        try:
                            imgs = call_controlnet_canny(
                                control_image=control_base,
                                prompt=prompt,
                                guidance_scale=guidance_scale,
                                steps=steps,
                                num_outputs=3,
                            )
                            st.session_state["controlled_images"] = imgs
                        except Exception as e:
                            st.error(f"ControlNet generation failed: {e}")
        else:
            st.info("Structural priors not available.")
    else:
        st.info("Upload Image B to compute structural priors and run ControlNet.")

    # ------------------ 4) Show controlled images at the bottom ------------------
    st.markdown("---")
    st.subheader("④ Controlled images (ControlNet output)")

    if st.session_state["controlled_images"]:
        cols2 = st.columns(3)
        for col, img in zip(cols2, st.session_state["controlled_images"]):
            with col:
                st.image(img, width=256)
        st.caption(
            "These images are generated by the ControlNet-Canny model on Replicate, "
            "conditioned on the chosen structural prior from Image B. "
            "Compare them to the uncontrolled SD img2img outputs above."
        )
    else:
        st.info(
            "No controlled images yet. Choose a structural prior above and click the "
            "ControlNet generate button."
        )


if __name__ == "__main__":
    main()

