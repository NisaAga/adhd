# Data Contract: Student 1 -> Student 2

## Files
- `data/processed/adhd_eeg_paper_protocol.npz`      full resolution, T = 11520 (256 Hz) - as in the paper
- `data/processed/adhd_eeg_paper_protocol_ds8.npz`  downsampled x8, T = 1440 (32 Hz) - ~8x faster (documented deviation)

## Loading
```python
from src.pipeline import load_processed
d = load_processed("data/processed/adhd_eeg_paper_protocol_ds8.npz")
```

| Key | Shape | Meaning |
|---|---|---|
| X_train, y_train | (79, T, 2), (79,) | cell 4  - channels (Cz, F4) |
| X_val, y_val | (79, T, 2) | cell 10 - **identical to cell 4** (see DATASET_NOTES) |
| X_test, y_test | (79, T, 2) | cell 11 - channels (Fz, F4) |
| X_eval, y_eval | (158, T, 2) | cell 10 + 11 -> **use for paper-comparable metrics** |
| fold | (79,) int 0-4 | stratified subject-wise fold id (same subject order in cell 4 and cell 11) |
| ids_*, groups_* | (N,) str | subject id e.g. `MC_17`, group FADHD/MADHD/FC/MC |
| channels_*, fs | | electrode names; sampling rate |

Labels: 0 = Healthy, 1 = ADHD. Data are float32, per-subject per-channel z-scored.

## Multi-head input
```python
inputs_train = [d["X_train"][:, :, 0:1], d["X_train"][:, :, 1:2]]   # head 1 = channel 1, head 2 = channel 2
inputs_eval  = [d["X_eval"][:, :, 0:1],  d["X_eval"][:, :, 1:2]]
```
Output: Dense(2, softmax) with `sparse_categorical_crossentropy`; P(ADHD) = `pred[:, 1]`.
Baselines use the whole `(N, T, 2)` tensor.

## Protocol 1 - reproduce the paper
Train on `X_train/y_train`; evaluate on `X_eval/y_eval` (158 samples). Use `X_val` for early stopping only if you
accept that it equals the training data (the paper did the same).

## Protocol 2 - honest estimate (report this too)
```python
for k in range(5):
    tr = d["fold"] != k; te = d["fold"] == k
    train = d["X_train"][tr]; y_tr = d["y_train"][tr]
    test  = np.concatenate([d["X_train"][te], d["X_test"][te]]); y_te = np.concatenate([d["y_train"][te]]*2)
```
Subjects in the test fold never appear in training.

## Metrics & plots (written by Student 1)
```python
from src.metrics import compute_metrics            # ACC, Prec, Rec, F1 (%), AUC, TP/TN/FP/FN - paper Eq. 1-5
from src import plots
m = compute_metrics(y_eval, p_adhd)
plots.confusion_matrix_plot(y_eval, (p_adhd>=.5).astype(int), "LSTM+BiLSTM", "cm_lstm_bilstm.png")
plots.roc_plot(y_eval, p_adhd, "LSTM+BiLSTM", "roc_lstm_bilstm.png")
plots.paper_comparison({"LSTM,BI-LSTM": m["ACC"], ...})      # bar chart vs paper
```
Paper targets: `plots.PAPER_MULTI`, `plots.PAPER_BASE`.

## Practical tip
11,520-step LSTMs are slow/memory hungry on Colab's K80/T4. Start with the `ds8` file, then try full resolution.
