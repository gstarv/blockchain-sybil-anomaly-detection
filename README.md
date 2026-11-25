# Anomalous Transaction & Sybil Behavior Detection in Blockchain Networks

An explainable machine learning pipeline for detecting suspicious wallet clustering, Sybil behavior, and abnormal transaction bursts on decentralized ledgers.

## Features

- Synthetic, configurable blockchain transaction dataset
- Isolation Forest unsupervised anomaly detection
- Random Forest classifier with SMOTE class balancing
- SHAP feature attributions
- Command-line options for data size and random seed
- Automated tests and CI

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python anomaly_detection.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Project structure

- `anomaly_detection.py` - synthetic data generation and the detection pipeline
- `tests/test_anomaly_detection.py` - unit tests
- `requirements.txt` - runtime dependencies

## Example

```bash
python anomaly_detection.py --normal 4750 --sybil 250 --seed 42
```

## Notes

The generated ledger is synthetic. It demonstrates the workflow but should not be interpreted as evidence that a real wallet is malicious.
