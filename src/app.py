from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import configure_logging
from src.ocr import create_ocr_engine
from src.pipeline import scan_image


configure_logging()
logger = logging.getLogger(__name__)


@st.cache_resource
def _get_ocr_engine():
    return create_ocr_engine()


def main() -> None:
    st.set_page_config(page_title="Whiteboard Reader", layout="wide")
    st.title("Whiteboard Reader")
    st.caption("Perspective correction, image enhancement, and pretrained OCR")

    uploaded = st.file_uploader("Choose a whiteboard image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded is None:
        return

    st.image(uploaded.getvalue(), caption="Uploaded image", use_container_width=True)
    if not st.button("Process image", type="primary"):
        return

    try:
        with st.spinner("Correcting and reading the image..."):
            result = scan_image(uploaded.getvalue(), engine=_get_ocr_engine())
    except Exception as error:
        logger.exception("Image processing failed")
        st.error(f"Unable to process this image: {error}")
        return

    if result.image.whiteboard_detected:
        st.success("A board-shaped region was detected and perspective-corrected.")
    else:
        st.info("No reliable board boundary was found; the full image was processed.")

    image_column, text_column = st.columns(2)
    with image_column:
        st.subheader("Enhanced image")
        st.image(result.image.enhanced, use_container_width=True)
    with text_column:
        st.subheader("Extracted text")
        st.text_area("OCR output", value=result.ocr.text, height=360, label_visibility="collapsed")
        if result.ocr.mean_confidence is not None:
            st.metric("Mean OCR confidence", f"{result.ocr.mean_confidence:.1%}")
        st.download_button(
            "Download text",
            data=result.ocr.text,
            file_name="whiteboard.txt",
            mime="text/plain",
            disabled=not result.ocr.text,
        )


if __name__ == "__main__":
    main()