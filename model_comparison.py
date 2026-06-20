import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

print("Loading dataset...")
df = pd.read_csv("final_balanced_dataset.csv")

batches = sorted(df["batch"].unique())
train_batch = batches[0]       # Batch 1
test_batches = batches[1:]     # Batches 2–10

# ── Models to benchmark ──────────────────────────────────────────────────────
models = {
    # ---- From Table 3 (reimplemented as baselines) ----
    "SVM-RBF":       SVC(kernel="rbf", C=1.0),
    "LDA":           LinearDiscriminantAnalysis(),
    "PCA-SVM":       Pipeline([("pca", PCA(n_components=0.95)), ("svm", SVC(kernel="rbf"))]),
    "SVM-GFK":       SVC(kernel="rbf", C=10.0),   # approximate

    # ---- Additional classical ML ----
    "KNN-5":         KNeighborsClassifier(n_neighbors=5),
    "KNN-3":         KNeighborsClassifier(n_neighbors=3),
    "NaiveBayes":    GaussianNB(),
    "DecisionTree":  DecisionTreeClassifier(random_state=42),
    "LogisticReg":   LogisticRegression(max_iter=1000, random_state=42),
    "RidgeClf":      RidgeClassifier(),

    # ---- Ensemble methods ----
    "RandomForest":     RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees":       ExtraTreesClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":         AdaBoostClassifier(n_estimators=100, random_state=42),
    "Bagging-SVM":      BaggingClassifier(estimator=SVC(kernel="rbf"),
                                          n_estimators=10, random_state=42),

    # ---- Boosting libraries ----
    "XGBoost":    xgb.XGBClassifier(n_estimators=200, use_label_encoder=False,
                                     eval_metric="mlogloss", random_state=42),
    "LightGBM":   lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    "CatBoost":   CatBoostClassifier(iterations=200, random_seed=42, verbose=0),

    # ---- Neural networks ----
    "MLP-small":  MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42),
    "MLP-deep":   MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42),
}

# ── Prepare train set ────────────────────────────────────────────────────────
train_df = df[df["batch"] == train_batch]
X_train = train_df.drop(["label", "batch"], axis=1).values
y_train = train_df["label"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ── Evaluate each model on every test batch ──────────────────────────────────
print("\n=== MODEL ACCURACY PER BATCH ===\n")
results = {}

for name, model in models.items():
    row = {}
    try:
        model.fit(X_train_scaled, y_train)
        accs = []
        for b in test_batches:
            b_df = df[df["batch"] == b]
            X_test = scaler.transform(b_df.drop(["label", "batch"], axis=1).values)
            y_test = b_df["label"].values
            acc = accuracy_score(y_test, model.predict(X_test))
            task_label = f"1-{b}"
            row[task_label] = round(acc, 4)
            accs.append(acc)
        row["Average"] = round(np.mean(accs), 4)
        results[name] = row
        print(f"{name:20s}  Avg: {row['Average']:.4f}")
    except Exception as e:
        print(f"{name:20s}  ERROR: {e}")

# ── Save results ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df = results_df.sort_values("Average", ascending=False)
results_df.to_csv("model_comparison.csv")

print("\n=== TOP 5 MODELS BY AVERAGE ACCURACY ===")
print(results_df["Average"].head(5).to_string())
print("\nFull results saved to model_comparison.csv")