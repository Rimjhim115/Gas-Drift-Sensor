# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")

# print("Loading dataset...")
# df = pd.read_csv("final_balanced_dataset.csv")

# batches = sorted(df["batch"].unique())
# train_batch = batches[0]   
# test_batches = batches[1:]

# train_df = df[df["batch"] == train_batch]
# X_train = train_df.drop(["label", "batch"], axis=1).values
# y_train = train_df["label"].values

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train_scaled, y_train)


# print("\nTask     | Accuracy")
# print("-" * 22)

# accs = []
# for b in test_batches:
#     b_df = df[df["batch"] == b]
#     X_test = scaler.transform(b_df.drop(["label", "batch"], axis=1).values)
#     y_test = b_df["label"].values
#     acc = accuracy_score(y_test, model.predict(X_test))
#     accs.append(acc)
#     print(f"1-{b:<6}  | {acc:.4f}")

# print("-" * 22)
# print(f"Average  | {np.mean(accs):.4f}")

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                               ExtraTreesClassifier,
                               VotingClassifier)
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CORAL  (CORrelation ALignment — Sun & Saenko, ECCV 2016)
# Transforms source Xs so its covariance matches target Xt.
# Closed-form, no labels from target needed.
# ─────────────────────────────────────────────────────────────────────────────
def coral_transform(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(d)

    ds, Vs = np.linalg.eigh(Cs)
    dt, Vt = np.linalg.eigh(Ct)
    ds = np.maximum(ds, eps)
    dt = np.maximum(dt, eps)

    # whitening matrix for source, re-colouring matrix for target
    A = (Vs @ np.diag(1.0 / np.sqrt(ds)) @ Vs.T) @ \
        (Vt @ np.diag(np.sqrt(dt))        @ Vt.T)
    return Xs @ A


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def build_ensemble():
    return VotingClassifier(
        estimators=[
            ("svc", SVC(kernel="rbf", C=50, gamma="scale",
                        probability=True, random_state=42)),
            ("rf",  RandomForestClassifier(n_estimators=500, random_state=42)),
            ("etc", ExtraTreesClassifier(n_estimators=500,  random_state=42)),
        ],
        voting="soft",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(csv_path: str):
    print("=" * 60)
    print(" Cross-Batch Smell Classification")
    print(" Train: Batch 1 only | Test: Batches 2-10")
    print(" Method: CORAL + Soft-Voting (SVC + RF + ETC)")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("label", "batch")]
    all_batches  = sorted(df["batch"].unique())
    test_batches = [b for b in all_batches if b != 1]

    train_df  = df[df["batch"] == 1]
    X_tr_raw  = train_df[feature_cols].values
    y_tr      = train_df["label"].values

    print(f"\nFeatures : {len(feature_cols)}")
    print(f"Train    : {len(X_tr_raw)} samples (batch 1)")
    print(f"Test     : batches {test_batches}\n")

    results   = []
    all_preds = []
    all_true  = []

    for b in test_batches:
        te_df    = df[df["batch"] == b]
        X_te_raw = te_df[feature_cols].values
        y_te     = te_df["label"].values

        # Step 1 — independent per-batch z-score
        Xtr = StandardScaler().fit_transform(X_tr_raw)
        Xte = StandardScaler().fit_transform(X_te_raw)

        # Step 2 — CORAL: align training covariance to test covariance
        Xtr = coral_transform(Xtr, Xte)

        # Step 3 — fit ensemble on adapted training data
        model = build_ensemble()
        model.fit(Xtr, y_tr)

        # Step 4 — predict on (z-scored) test data
        preds = model.predict(Xte)
        acc   = accuracy_score(y_te, preds)
        results.append((b, acc))
        all_preds.extend(preds)
        all_true.extend(y_te)
        print(f"  1 -> {b:<3}  accuracy = {acc:.4f}")

    avg = np.mean([a for _, a in results])
    print(f"\n  Average   accuracy = {avg:.4f}")

    print("\n" + "-" * 60)
    print(" Overall classification report (batches 2-10 combined)")
    print("-" * 60)
    labels = sorted(np.unique(all_true))
    print(classification_report(
        all_true, all_preds,
        labels=labels,
        target_names=[f"class{l}" for l in labels]
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="final_balanced_dataset.csv",
                        help="Path to the dataset CSV")
    args = parser.parse_args()
    main(args.csv)