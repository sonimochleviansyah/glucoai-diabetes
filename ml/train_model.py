"""Pelatihan model Logistic Regression untuk klasifikasi risiko diabetes (Pima Indians).
Jalankan: python train_model.py
Output: model.joblib, reports/metrics.json, reports/*.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv("data/diabetes.csv")
print("Ukuran data:", df.shape)
print("Distribusi kelas:\n", df["Outcome"].value_counts())

X = df.drop(columns="Outcome")
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def build_pipeline(impute=True):
    steps = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps += [("scaler", StandardScaler()),
              ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                         random_state=RANDOM_STATE))]
    return Pipeline(steps)


def evaluate(model, Xt, yt):
    pred = model.predict(Xt)
    proba = model.predict_proba(Xt)[:, 1]
    return {"accuracy": accuracy_score(yt, pred),
            "precision": precision_score(yt, pred),
            "recall": recall_score(yt, pred),
            "f1": f1_score(yt, pred),
            "roc_auc": roc_auc_score(yt, proba)}


# Skenario A: nilai 0 dibiarkan apa adanya (tanpa penanganan data hilang)
model_a = build_pipeline(impute=False).fit(X_train, y_train)
res_a = evaluate(model_a, X_test, y_test)

# Skenario B: nilai 0 pada 5 kolom dianggap data hilang lalu diimputasi median
def zeros_to_nan(frame):
    frame = frame.copy()
    frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
    return frame

Xtr_b, Xte_b = zeros_to_nan(X_train), zeros_to_nan(X_test)
model_b = build_pipeline(impute=True).fit(Xtr_b, y_train)
res_b = evaluate(model_b, Xte_b, y_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(build_pipeline(True), zeros_to_nan(X), y, cv=cv, scoring="f1")

print("\nA (tanpa penanganan):", {k: round(v, 4) for k, v in res_a.items()})
print("B (imputasi median) :", {k: round(v, 4) for k, v in res_b.items()})
print("CV F1 5-fold (B)    :", cv_scores.round(4), "rata-rata", cv_scores.mean().round(4))

# Visualisasi
pred_b = model_b.predict(Xte_b)
cm = confusion_matrix(y_test, pred_b)
ConfusionMatrixDisplay(cm, display_labels=["Tidak Diabetes", "Diabetes"]).plot(cmap="Blues")
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_test, model_b.predict_proba(Xte_b)[:, 1])
plt.plot(fpr, tpr, label=f"AUC = {res_b['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Kurva ROC"); plt.legend(); plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150); plt.close()

coef = pd.Series(model_b.named_steps["clf"].coef_[0], index=X.columns).sort_values()
coef.plot(kind="barh", color="steelblue")
plt.title("Koefisien Logistic Regression (fitur terstandarisasi)")
plt.tight_layout(); plt.savefig("reports/coefficients.png", dpi=150); plt.close()

# Simpan model final (skenario B) dan metrik
joblib.dump(model_b, "model.joblib")
metrics = {
    "n_rows": int(len(df)),
    "class_counts": {int(k): int(v) for k, v in y.value_counts().items()},
    "scenario_A_no_missing_handling": res_a,
    "scenario_B_median_imputation": res_b,
    "confusion_matrix": cm.tolist(),
    "cv_f1_scores": cv_scores.tolist(),
    "cv_f1_mean": float(cv_scores.mean()),
    "coefficients": coef.to_dict(),
    "feature_medians": {c: float(X[c].replace(0, np.nan).median())
                        if c in ZERO_AS_MISSING else float(X[c].median())
                        for c in X.columns},
}
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nModel dan laporan tersimpan.")
