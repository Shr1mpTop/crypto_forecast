"""Compute Public/Private/Final scores locally for SC6117.

Usage:
  python score_submission.py submissions/timestamp_prediction_submission.csv
  # or score all csv under submissions/
  python score_submission.py

Requirements:
- Submission must have 2881 rows.
- Columns: either ['row_id','Target'] or ['Timestamp','Prediction'] or a second column with predictions.
"""

import sys
import glob
import pandas as pd
import numpy as np

NEXT_CLOSE_AFTER_LAST = 84284.01  # close at 2025-11-22 23:45
TEST_PATH = 'data/test.csv'
SPLIT_INDEX = None  # will be set from test length (50/50)


def load_y_true():
    test = pd.read_csv(TEST_PATH)
    n = len(test)
    split = n // 2
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE_AFTER_LAST
    y_true = np.log(next_close / test['Close']).values
    return y_true, split, n


def read_pred(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError('empty submission')
    if len(df) != expected_len:
        raise ValueError(f'row count {len(df)} != expected {expected_len}')

    # detect prediction column
    for col in ['Target', 'Prediction', 'target', 'prediction']:
        if col in df.columns:
            return df[col].to_numpy()
    # fallback to second column
    if df.shape[1] >= 2:
        return df.iloc[:, 1].to_numpy()
    raise ValueError('no prediction column found')


def score(pred: np.ndarray, true: np.ndarray, split: int):
    pub = np.corrcoef(pred[:split], true[:split])[0, 1]
    priv = np.corrcoef(pred[split:], true[split:])[0, 1]
    final = 0.5 * pub + 0.5 * priv
    return pub, priv, final


def process(path: str, y_true: np.ndarray, split: int):
    try:
        pred = read_pred(path)
        pub, priv, final = score(pred, y_true, split)
        name = path.replace('\\', '/').split('/')[-1]
        print(f"{name:<40} Public={pub:>8.5f} Private={priv:>8.5f} Final={final:>8.5f}")
    except Exception as e:
        print(f"{path}: ERROR -> {e}")


if __name__ == '__main__':
    y_true, split, expected_len = load_y_true()
    print(f"Test rows: {expected_len}, Public={split}, Private={expected_len - split}")
    print(f"Reading submissions from args or submissions/*.csv\n")

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = glob.glob('submissions/*.csv')

    for p in paths:
        process(p, y_true, split)
