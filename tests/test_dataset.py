import cv2
import numpy as np

from src.dataset import scan_dataset


def test_scan_pairs_image_and_transcript_without_touching_sources(tmp_path) -> None:
    course_dir = tmp_path / "course1"
    course_dir.mkdir()
    image_path = course_dir / "board.jpg"
    transcript_path = course_dir / "notes.txt"
    cv2.imwrite(str(image_path), np.full((60, 80, 3), 255, dtype=np.uint8))
    transcript_path.write_text("Known transcript\n", encoding="utf-8")
    before = (image_path.read_bytes(), transcript_path.read_bytes())

    scan = scan_dataset(tmp_path)

    assert len(scan.records) == 1
    assert scan.records[0].valid
    assert scan.records[0].transcript == "Known transcript"
    assert (image_path.read_bytes(), transcript_path.read_bytes()) == before


def test_scan_reports_ambiguous_folder(tmp_path) -> None:
    course_dir = tmp_path / "course2"
    course_dir.mkdir()
    (course_dir / "one.jpg").write_bytes(b"invalid")
    (course_dir / "two.jpg").write_bytes(b"invalid")
    (course_dir / "notes.txt").write_text("text", encoding="utf-8")

    scan = scan_dataset(tmp_path)

    assert not scan.records
    assert "expected one image and one transcript" in scan.issues[0]