# Evidence and image provenance

This folder contains **documentation evidence** for the TechCrush Cohort 8 Handwritten Digit Recognizer. These are photographs, app screenshots and figures already created during the project—not newly synthesized handwriting or simulated app results.

## Original real-world examples

| Repo asset | Source file supplied during development | What it actually shows |
|---|---|---|
| `images/photos/handwritten-9254.jpg` | `20261015_021126.jpg` | Original blue-ink `9254` on ruled paper; used during a real segmentation-debugging iteration. |
| `images/screenshots/phone-9254-success.jpg` | `Screenshot_20261015-024437_Chrome.jpg` | Streamlit screenshot reporting `9254`, with four actual detection boxes and 28×28 region previews. |
| `images/photos/handwritten-347.jpg` | `20261015_105354.jpg` | Original blue-ink `347` on ruled paper. |
| `images/screenshots/phone-347-success.jpg` | `Screenshot_20261015-113455_Chrome.jpg` | Streamlit reporting `347`, three detected regions and three actual digit previews. |
| `images/photos/handwritten-962.jpg` | `20261015_105359.jpg` | Original `962`; an earlier manual app test classified it as `961`. Included as a challenging photo, not a verified success. |

## Additional phone-camera recognition captures

| Repo asset | Original browser capture | Observed output |
|---|---|---|
| `images/screenshots/photo-912-success.jpg` | `WhatsApp Image 2026-10-15 at 17.06.40.jpeg` | Real handwritten `912` photographed on ruled paper; app reported `912` with three detected boxes. |
| `images/screenshots/photo-268-success.jpg` | `WhatsApp Image 2026-10-15 at 17.06.39.jpeg` | Real handwritten `268` photographed on ruled paper; app reported `268` with three detected boxes. |
| `images/screenshots/photo-543-success.jpg` | `WhatsApp Image 2026-10-15 at 17.06.39 (1).jpeg` | Real handwritten `543` photographed on ruled paper; app reported `543` with three detected boxes. |

These three screenshots were cropped to remove phone/browser chrome without changing the displayed handwriting or model outputs. They are **single observed trials**, not a held-out performance dataset.

## Actual application screenshots

| Repo asset | Source file | Interpretation |
|---|---|---|
| `images/screenshots/canvas-72341-success.jpg` | `Screenshot 2026-10-15 175043.png` | Actual five-digit canvas example recognized as `72341`, with five detection boxes. |
| `images/screenshots/canvas-674-misclassified.jpg` | `Screenshot 2026-10-15 170725.png` | Actual three-region canvas `674` test predicted as `639`. This is an error case, not a successful recognition. |
| `images/screenshots/single-digit-7-correct.jpg` | `Screenshot 2026-10-15 171049.png` | Actual single-digit canvas `7` predicted as `7`. |
| `images/screenshots/single-digit-4-misclassified.jpg` | `Screenshot 2026-10-15 171209.png` | Actual handwritten `4` classified as `6` with high softmax score. |

## Actual model experiment figures

| Repo asset | Notebook origin | Interpretation |
|---|---|---|
| `images/experiments/baseline-vs-improved-test-accuracy.png` | `HWDR (3).ipynb`, code cell index 228, saved chart output | Baseline 98.67% vs selected improved CNN 99.57% on MNIST's 10,000-image test set. |
| `images/experiments/improved-cnn-confusion-matrix.png` | `HWDR (3).ipynb`, code cell index 241, saved chart output | Final improved CNN test-set confusion matrix; not a real-world photo benchmark. |

## Processing applied to the documentation copies

- Original camera photographs were EXIF-corrected where necessary, resized to fit within 1400 × 1400 pixels, re-encoded to JPEG and stripped of EXIF metadata for the public README. Their written characters were not edited.
- Application screenshots were either copied/re-encoded or cropped to remove phone chrome and unrelated UI space. Their displayed model outputs and detection boxes were not recreated or changed.
- Notebook plots were exported from saved `image/png` outputs without replotting.
- No AI-generated test images were used in the README visuals.

## Evidence limitations

A screenshot demonstrates one specific application run. It does not quantify generalization, repeatability or real-world accuracy across varied writers. Some screenshots date from intermediate versions, so an issue in an old screenshot may have been addressed later. The README identifies such cases as historical observations.

The user later reported successful retests `14865 -> 14865` (expected count five) and `4234 -> 4234` (Automatic); screenshots of those exact successful retests were not included in this folder. They are labeled as user-reported, not visually archived results.

The separate regression-test fixtures under `tests/fixtures/` are not automatically identical to these human-facing docs copies and should be maintained independently if the tests use specific pixel coordinates.
