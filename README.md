# ADHD EEG Classification - Data Engineering Module (Student 1)

Reproduction of *"Comparative study of multi-headed and baseline deep learning models for ADHD classification
from EEG signals"* (Amar et al., 2025). This module covers: load MAT files -> extract cells 4/10/11 -> labels ->
normalisation -> train/val/test datasets -> plots & confusion matrices. Student 2 builds the models.

## Folder structure
```
adhd_eeg_project/
├── data/raw/            <- put FADHD.mat, MADHD.mat, FC.mat, MC.mat here
├── data/processed/      <- generated .npz (hand this to Student 2)
├── src/
│   ├── config.py        paths, cells, labels, preprocessing switches
│   ├── data_loader.py   MAT (v5 + v7.3) -> NumPy
│   ├── dataset_builder.py  stack groups, labels, subject ids
│   ├── preprocessing.py   z-score, crop, downsample, quality checks
│   ├── pipeline.py      end-to-end + load_processed() for Student 2
│   ├── metrics.py       ACC/Prec/Rec/F1/AUC exactly as paper Eq. 1-5
│   └── plots.py         EDA, confusion matrix, ROC, paper-vs-ours charts
├── scripts/             00 synthetic data, 01 inspect, 02 build, 03 EDA plots
├── tests/test_pipeline.py
├── docs/                DATASET_NOTES, HANDOFF_TO_STUDENT2, DATA_ENGINEERING_REPORT
└── outputs/figures, outputs/reports
```

## Run order (real data already placed in data/raw/)
```bash
pip install -r requirements.txt
python scripts/01_inspect_mat.py              # real shapes of all 44 cells
python scripts/02_build_dataset.py            # full resolution  -> data/processed/adhd_eeg_paper_protocol.npz
python scripts/02_build_dataset.py --downsample 8   # fast version (T=1440)
python scripts/03_eda_plots.py                # figures -> outputs/figures/
python -m pytest tests                        # unit tests (synthetic data)
```
`scripts/00_make_synthetic_data.py` only generates fake `.mat` files for testing - never use it for results.


##vedika's part
### Google Colab
Upload the folder to Drive, then:
```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/adhd_eeg_project
!pip install -q -r requirements.txt
!python scripts/01_inspect_mat.py
```

## Key findings on the real data (details: docs/DATASET_NOTES.md)
- 79 subjects after removing `FADHD_06` (dead F4 channel): 37 ADHD / 42 healthy. Cells 4/10/11 have 11,520 samples.
- **Cell 10 is byte-identical to cell 4** -> the paper's "validation" set equals its training set. Paper-comparable
  metrics use `X_eval` (cells 10+11, 158 samples), but we also export a leak-free subject-wise 5-fold protocol.
- Hand-off to Student 2: `docs/HANDOFF_TO_STUDENT2.md`. Report: `docs/DATA_ENGINEERING_REPORT.md`.
