import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly_detection import FEATURE_COLUMNS, generate_synthetic_ledger_data


def test_synthetic_dataset_shape():
    df = generate_synthetic_ledger_data(n_normal=40, n_sybil=10, seed=7)
    assert len(df) == 50
    assert set(df["is_anomalous"]) == {0, 1}
    assert list(df.columns) == FEATURE_COLUMNS + ["is_anomalous"]


def test_invalid_dataset_sizes_are_rejected():
    try:
        generate_synthetic_ledger_data(n_normal=1, n_sybil=1)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
