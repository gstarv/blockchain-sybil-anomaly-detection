import argparse
import numpy as np
import pandas as npd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap

FEATURE_COLUMNS = [
    "tx_frequency_per_hr",
    "mean_inter_arrival_sec",
    "in_degree_centrality",
    "out_degree_centrality",
    "value_variance_ratio",
    "unique_counterparties",
    "burstiness_index",
]


def generate_synthetic_ledger_data(n_normal: int = 4750, n_sybil: int = 250, seed: int = 42):
    """Generate a reproducible, imbalanced synthetic wallet dataset."""
    if n_normal < 2 or n_sybil < 1:
        raise ValueError("n_normal must be at least 2 and n_sybil at least 1")

    rng = np.random.default_rng(seed)
    normal = npd.DataFrame({
        "tx_frequency_per_hr": rng.gamma(2.0, 1.5, n_normal),
        "mean_inter_arrival_sec": rng.lognormal(6.5, 1.1, n_normal),
        "in_degree_centrality": rng.beta(2.0, 10.0, n_normal),
        "out_degree_centrality": rng.beta(2.0, 10.0, n_normal),
        "value_variance_ratio": rng.uniform(0.1, 1.5, n_normal),
        "unique_counterparties": rng.poisson(12, n_normal),
        "burstiness_index": rng.normal(0.15, 0.1, n_normal),
        "is_anomalous": np.zeros(n_normal, dtype=int),
    })

    sybil = npd.DataFrame({
        "tx_frequency_per_hr": rng.gamma(8.0, 4.5, n_sybil),
        "mean_inter_arrival_sec": rng.lognormal(2.2, 0.6, n_sybil),
        "in_degree_centrality": rng.beta(7.0, 2.5, n_sybil),
        "out_degree_centrality": rng.beta(6.5, 3.0, n_sybil),
        "value_variance_ratio": rng.uniform(0.001, 0.08, n_sybil),
        "unique_counterparties": rng.poisson(3, n_sybil),
        "burstiness_index": rng.normal(0.78, 0.12, n_sybil),
        "is_anomalous": np.ones(n_sybil, dtype=int),
    })

    return npd.concat([normal, sybil], ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)


def run_pipeline(n_normal: int = 4750, n_sybil: int = 250, seed: int = 42) -> None:
    print("1. Loading behavioral blockchain transaction features...")
    df = generate_synthetic_ledger_data(n_normal, n_sybil, seed)
    X = df[FEATURE_COLUMNS]
    y = df["is_anomalous"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n2. Training Isolation Forest baseline...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=n_sybil / (n_normal + n_sybil),
        random_state=seed,
    )
    iso_forest.fit(X_train_scaled)
    iso_preds = np.where(iso_forest.predict(X_test_scaled) == -1, 1, 0)
    print(f"Isolation Forest F1-Score: {f1_score(y_test, iso_preds):.4f}")

    print("\n3. Balancing the minority class with SMOTE...")
    smote = SMOTE(random_state=seed)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    rf_clf.fit(X_train_bal, y_train_bal)

    rf_preds = rf_clf.predict(X_test_scaled)
    rf_probs = rf_clf.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Random Forest Evaluation ---")
    print(classification_report(y_test, rf_preds, digits=4, zero_division=0))
    print(f"Macro F1-Score : {f1_score(y_test, rf_preds, average='macro'):.4f}")
    print(f"ROC-AUC Score  : {roc_auc_score(y_test, rf_probs):.4f}")

    print("\n4. Computing SHAP feature attributions...")
    explainer = shap.TreeExplainer(rf_clf)
    values = explainer.shap_values(X_test_scaled)

    if isinstance(values, list):
        positive_values = values[1]
    elif getattr(values, "ndim", 0) == 3:
        positive_values = values[:, :, 1]
    else:
        positive_values = values

    importance = np.abs(positive_values).mean(axis=0)
    importance_df = npd.DataFrame({
        "Feature": X.columns,
        "Mean_Abs_SHAP": importance,
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    print("\n--- Top Risk Drivers ---")
    print(importance_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Run blockchain anomaly detection simulation")
    parser.add_argument("--normal", type=int, default=4750, help="Number of normal wallets")
    parser.add_argument("--sybil", type=int, default=250, help="Number of Sybil wallets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.normal, args.sybil, args.seed)
