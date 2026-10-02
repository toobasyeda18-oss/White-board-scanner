import cv2
import numpy as np
import pytest

from src.preprocessing import load_image, preprocess_image


def test_preprocess_returns_enhanced_and_binary_variants() -> None:
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (25, 20), (295, 220), (30, 30, 30), 3)
    cv2.putText(image, "Board notes", (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    processed = preprocess_image(image)

    assert processed.enhanced.ndim == 2
    assert processed.binary.shape == processed.enhanced.shape
    assert processed.corrected.size > 0


def test_load_image_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="not a readable"):
        load_image(b"not an image")


def test_preprocess_detects_large_quadrilateral() -> None:
    image = np.full((300, 400, 3), 25, dtype=np.uint8)
    corners = np.array([[60, 40], [340, 55], [330, 255], [50, 240]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (245, 245, 245))
    cv2.polylines(image, [corners], True, (10, 10, 10), 4)

    processed = preprocess_image(image)

    assert processed.whiteboard_detected
    assert processed.corrected.shape[0] < image.shape[0]
    assert processed.corrected.shape[1] < image.shape[1]