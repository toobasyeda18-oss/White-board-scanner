from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    original: np.ndarray
    corrected: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    whiteboard_detected: bool


def load_image(source: bytes | bytearray | Path | str) -> np.ndarray:
    try:
        image_bytes = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    except OSError as error:
        raise ValueError(f"Unable to read image: {error}") from error

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("The file is not a readable supported image.")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _find_board_corners(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    scale = min(1.0, 1400 / max(height, width))
    if scale < 1.0:
        search_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        search_image = image

    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    search_area = float(search_image.shape[0] * search_image.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.02, 0.03, 0.04, 0.06):
            polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            area = float(cv2.contourArea(polygon))
            if area >= search_area * 0.12 and area < search_area * 0.97:
                candidates.append((area, polygon.reshape(4, 2).astype(np.float32)))
                break

    if not candidates:
        return None

    corners = _order_points(max(candidates, key=lambda candidate: candidate[0])[1])
    if scale < 1.0:
        corners /= scale
    if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= width):
        return None
    if np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= height):
        return None
    return corners


def _rectify(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    width = max(width, 2)
    height = max(height, 2)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def preprocess_image(source: bytes | bytearray | Path | str | np.ndarray) -> ProcessedImage:
    original = source.copy() if isinstance(source, np.ndarray) else load_image(source)
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("Expected a three-channel color image.")

    corners = _find_board_corners(original)
    detected = corners is not None
    corrected = _rectify(original, corners) if corners is not None else original.copy()
    if not detected:
        logger.info("No board quadrilateral found; retaining the full image.")

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return ProcessedImage(original, corrected, enhanced, binary, detected)