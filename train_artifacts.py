from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Script reproduksi artifact model.
# Catatan: script ini memakai fitur hasil GA dan parameter terbaik Bayesian Optimization dari notebook.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "PCOS_data_without_infertility.xlsx"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "PCOS (Y/N)"
SHEET_NAME = "Full_new"
DROP_COLS = ["Sl. No", "Patient File No.", "Unnamed: 44"]
SELECTED_FEATURES = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))["selected_features"]
BEST_C = 0.005461850839311912

raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
raw.columns = raw.columns.str.strip()
df = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns], errors="ignore").copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

feature_columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

X_train_processed = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
X_test_processed = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)

X_train_final = X_train_processed[SELECTED_FEATURES]
X_test_final = X_test_processed[SELECTED_FEATURES]

model = SVC(kernel="linear", C=BEST_C, class_weight="balanced", random_state=42)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)
y_score = model.decision_function(X_test_final)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred, zero_division=0))
print("ROC-AUC  :", roc_auc_score(y_test, y_score))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Tidak PCOS", "PCOS"], zero_division=0))

joblib.dump(model, ARTIFACT_DIR / "svm_linear_bayes_ga_model.pkl")
joblib.dump(imputer, ARTIFACT_DIR / "median_imputer.pkl")
joblib.dump(scaler, ARTIFACT_DIR / "standard_scaler.pkl")
