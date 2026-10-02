# Whiteboard Reader Scanner

A small, reproducible computer-vision application that detects and rectifies a whiteboard-like quadrilateral, enhances its image, and extracts text with a pretrained OCR model. It does not train a model: the supplied dataset is too small and contains transcripts but no localization annotations or train/validation split.

## Dataset

The repository's supplied data is under `data/raw/archive (1)/data/`. It contains eight course folders, each with one JPEG and one UTF-8 text transcript: courses 1-4 and 6-9. Course 5 is absent. The transcript files in courses 7 and 8 have identical contents, which is reported as a data-quality warning rather than silently corrected. There are no boxes, polygons, class labels, or split definitions. The image/transcript association is based only on the files being together in the same course folder.

No dataset source URL or attribution was present in the supplied project files. Verify and add the original source citation before redistributing the dataset. `data/raw/` is not changed by this application and is excluded from Git.

## Features

- Recursive image/transcript inventory with image decode checks, empty-label checks, duplicate-label warnings, and a CSV/JSON data report.
- OpenCV preprocessing: edge/contour-based quadrilateral search, perspective warp when a candidate board is found, grayscale conversion, CLAHE contrast enhancement, and adaptive threshold output.
- Pretrained RapidOCR inference through ONNX Runtime; model weights are provided/downloaded by the dependency, not trained from this dataset.
- Optional labeled evaluation using word error rate (WER) and character error rate (CER); results are calculated from OCR predictions and the folder-local transcripts, never hard-coded.
- Streamlit upload interface with processed image, extracted text, confidence, and text download.
- Logging, validation errors, and tests built around synthetic inputs.

## Setup

Python 3.10 or newer is recommended. From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

RapidOCR's pretrained ONNX model files are bundled with its package, so no separate model download or external Tesseract installation is required.

## Pipeline

1. `src.dataset` pairs one supported image and one `.txt` transcript within each folder, validates readability, and reports ambiguous or incomplete folders.
2. `src.preprocessing` searches image edges for a large convex four-corner contour. If one is found, it rectifies the perspective; otherwise, it retains the full frame and reports that fallback. It then creates enhanced grayscale and adaptive-threshold variants.
3. `src.ocr` runs pretrained RapidOCR on the enhanced image.
4. `src.evaluation` compares prediction text with the supplied transcript using normalized WER and CER.

## Usage

Create a processed manifest, data-quality report, and enhanced image previews without modifying the originals:

```powershell
python -m src.prepare_data
```

Start the scanner:

```powershell
streamlit run src/app.py
```

Evaluate OCR against the available transcripts

```powershell
python -m src.evaluation
```

Run the tests:

```powershell
python -m pytest
```

Generated files go to `data/processed/` and are ignored by Git. OCR evaluation can take time on the first run while the pretrained model initializes. To evaluate a different dataset directory, use `python -m src.prepare_data --help` or `python -m src.evaluation --help`.

## Evaluation

WER counts word substitutions, deletions, and insertions relative to transcript words; CER does the same over normalized characters. The evaluation command reports macro averages across successfully evaluated image/transcript pairs and the number skipped. On the supplied eight pairs, one local run evaluated all 8 with 0 skipped and measured mean WER 0.9262 and mean CER 0.7411. These high error rates indicate that this pretrained general-text OCR baseline performs poorly on the supplied pairs. The repeated course 7/8 transcript and the small dataset can bias the result. Since no independent ground-truth verification is supplied, these metrics describe agreement with the text files, not general whiteboard-reading accuracy. Run the evaluation command to reproduce the scores in your environment.

## Technologies

Python, OpenCV, RapidOCR/ONNX Runtime, Streamlit, and pytest.
