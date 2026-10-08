# Data Engineering Report - ADHD EEG Classification (Student 1)

## 1. Objective
Prepare the Mendeley ADHD EEG dataset for reproducing the multi-headed LSTM/BiLSTM/GRU models of
Amar et al. (2025) (target: LSTM+BiLSTM, 89.87% accuracy).

## 2. Dataset
Mendeley Data (DOI 10.17632/6k4g25fhzg.1): resting-state / cognitive / sound-listening EEG, 256 Hz, 2 channels per cell.
Four MAT files (FADHD, MADHD, FC, MC), each a 1x11 cell array of (subjects x samples x channels).
Paper cohort: 79 adults (37 ADHD, 42 healthy). Experiment cells: 4 (train), 10 (validation), 11 (test).

## 3. Pipeline (code: `src/`, entry points: `scripts/`)
1. **Load** MAT files (SciPy; HDF5 fallback) -> NumPy.
2. **Audit**: detect NaN / constant channels, check duplicate cells, verify subject counts.
3. **Clean**: remove `FADHD_06` (dead F4 channel) -> 79 subjects.
4. **Label & assemble**: FADHD+MADHD = 1, FC+MC = 0; stable subject ids.
5. **Length**: cells 4/10/11 all have 11,520 samples (45 s) - no cropping needed; optional downsample (x8 -> 32 Hz).
6. **Normalise**: per-subject, per-channel z-score.
7. **Splits**: train = cell 4; val = cell 10; test = cell 11; eval = cell 10+11 (158); plus 5 stratified subject-wise folds.
8. **Export**: `.npz` + `outputs/reports/pipeline_report*.json`.

## 4. Results of data preparation
| Split | Cell | Channels | Shape (full) | ADHD | Healthy |
|---|---|---|---|---|---|
| Train | 4 | Cz, F4 | 79 x 11520 x 2 | 37 | 42 |
| Val | 10 | (= cell 4) | 79 x 11520 x 2 | 37 | 42 |
| Test | 11 | Fz, F4 | 79 x 11520 x 2 | 37 | 42 |
| Eval | 10+11 | - | 158 x 11520 x 2 | 74 | 84 |

Folds: sizes 16/16/16/16/15 subjects. Figures in `outputs/figures/` (class balance, sample signals, mean PSD).
Spectra show a ~10 Hz alpha peak and relatively higher low-frequency (delta/theta) power in ADHD in the cognitive-task
recording (cell 4), after per-subject normalisation.

## 5. Key findings / deviations from the paper
1. FADHD contains 11 subjects; one has a flat F4 channel -> excluded to match the paper's 79.
2. **Cell 10 is identical to cell 4** in all four files -> the paper's validation set is its training set; half of the
   158-sample evaluation set is training data. Reported accuracy is therefore optimistic.
3. Signals are 11,520 samples (45 s), not 7,680 as in the paper's example.
4. The paper gives no normalisation; z-score is our choice (configurable).
5. We additionally provide a leak-free subject-wise 5-fold protocol.

## 6. Limitations
Small cohort (79); same subjects across cells; train/test channels differ (Cz vs Fz); no artefact rejection;
unspecified hyper-parameters.

## 7. Reproducibility
`python scripts/01_inspect_mat.py && python scripts/02_build_dataset.py && python scripts/03_eda_plots.py`
Seed 42 (fold assignment). Tests: `python -m pytest tests` (synthetic data).

## 8. References
Amar L.A., Otifi A.M., Mohamed S.A. (2025) Phys Eng Sci Med 48:1657-1665.
Sadeghi Bajestani G. et al. (2023) Mendeley Data V1, doi:10.17632/6k4g25fhzg.1.
