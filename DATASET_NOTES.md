# Dataset Audit & Findings (verified on the real Mendeley files)

Dataset: Sadeghi Bajestani et al. (2023), Mendeley Data V1, DOI 10.17632/6k4g25fhzg.1
Paper: Amar, Otifi, Mohamed (2025), *Phys Eng Sci Med* 48:1657-1665.

## 1. Real structure (from `scripts/01_inspect_mat.py`)
Each file = 1x11 cell array; each cell = (subjects, samples, channels=2), 256 Hz.

| Cell | Samples | Duration | Cell | Samples | Duration |
|---|---|---|---|---|---|
| 1 | 7680 | 30 s | 7 | 7680 | 30 s |
| 2, 3 | 5120 | 20 s | 8, 9 | 5120 | 20 s |
| **4** | **11520** | **45 s** | 5 | 3840 | 15 s |
| **10, 11** | **11520** | **45 s** | 6 | 7680 | 30 s |

-> Cells 4/10/11 have **T = 11,520** samples (not 7,680 as in the paper's example).

| File | Raw subjects | Used | Label |
|---|---|---|---|
| FADHD | **11** | 10 | ADHD (1) |
| MADHD | 27 | 27 | ADHD (1) |
| FC | 13 | 13 | Healthy (0) |
| MC | 29 | 29 | Healthy (0) |
| Total | 80 | **79** (37 ADHD / 42 healthy) | |

## 2. Finding A - one FADHD subject is unusable (11 -> 10)
`FADHD_06` (0-based index 6) has a **constant channel 2 (F4)**: std = 0, a single unique value, in cells 4, 10 and 11.
The paper reports 10 female ADHD participants (Table 2) and 79 in total, so dropping this subject reproduces the
paper's cohort exactly. It is listed in `config.EXCLUDE_SUBJECTS` and the pipeline verifies by automatic detection that
this is the *only* bad subject.

## 3. Finding B - Cell 10 is an exact copy of Cell 4  (IMPORTANT)
For all four files `np.array_equal(cell4, cell10) == True` (both channels, all subjects). The paper's Table 3 labels
cell 10 as "F3, F4 / eyes closed", but the file contains exactly the same numbers as cell 4 (Cz, F4 / cognitive challenge).
Consequences:
- The paper trains on cell 4 and "validates" on cell 10 -> **validation = training data**.
- The paper's confusion matrices sum to 158 = cell 10 (79) + cell 11 (79). Half of that evaluation set is data the model
  was trained on, which inflates accuracy. Cell 11 is the only genuinely different recording.
- Cell 11 is also the same 79 people, so even that half is not subject-independent.

This does not stop us from reproducing the paper's *procedure*, but the 89.87% should be presented as a number obtained
under a leaky protocol. We therefore export two protocols (Section 5).

## 4. Other verified facts
- No NaN/Inf values. Amplitude up to ~517 uV (artefacts); a few subjects have high variance (e.g. MADHD_24, std ~134).
  Per-subject z-scoring handles scale differences; no artefact rejection is applied (the paper describes none).
- Channel 2 is F4 in all three cells; channel 1 is Cz (cell 4/10) or Fz (cell 11) per the paper's Table 3.
  (Cell 10 contains Cz, not F3, because it duplicates cell 4.)
- Subject order is consistent across cells (ids like `MC_17` refer to the same index in every cell).
- Class balance 37/42 in every split; 158-sample eval set = 74 ADHD + 84 healthy (matches TP+FN=74, TN+FP=84 in Table 4).

## 5. Protocols exported
| Protocol | Train | Test | Purpose |
|---|---|---|---|
| **Paper** | `X_train` (cell 4) | `X_eval` (cell 10 + 11, n = 158) | reproduce Tables 4/5 |
| **Honest (recommended as 2nd result)** | cell 4 subjects with `fold != k` | cell 4 *and* cell 11 subjects with `fold == k` | subject-wise 5-fold CV, no leakage |
| Cell-11-only | `X_train` | `X_test` (cell 11) | train/test on different recordings |

## 6. Not specified in the paper (Student 2 decides)
Optimizer, learning rate, epochs, batch size, units, dropout, seed, threshold. Exact 89.87% may not be bit-reproducible.
Z-score normalisation is our addition (switch in `config.py`); the paper says raw signals.
