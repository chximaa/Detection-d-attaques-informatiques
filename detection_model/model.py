
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_FILE = "cyber_data_clean.xlsx"
TARGET = "is_attack"

# Reproducible results
RANDOM_STATE = 42

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_excel(DATA_FILE)

print("\n" + "=" * 65)
print("          CYBERSECURITY ML MODEL TRAINING")
print("=" * 65)

print(f"\nDataset shape: {df.shape}")
print(f"Target column: {TARGET}")

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. Available columns: "
        f"{df.columns.tolist()}"
    )

# Remove rows with a missing target
df = df.dropna(subset=[TARGET]).copy()

# The dataset includes a 'label' column.
# Exclude it to avoid potential target leakage.
excluded_columns = [TARGET]

if TARGET == "is_attack" and "label" in df.columns:
    excluded_columns.append("label")

X = df.drop(columns=excluded_columns)
y = df[TARGET]

if y.nunique() < 2:
    raise ValueError("The target must contain at least two classes.")

# This script is designed for classification
if y.nunique() > 20:
    raise ValueError(
        "The target has many distinct values. "
        "Check whether this is a classification target."
    )

# Convert to numeric features and replace infinite values
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Drop features that contain no usable information
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise ValueError("No usable feature columns were found.")

print(f"Samples: {len(X)}")
print(f"Features used: {X.shape[1]}")
print(f"Excluded columns: {excluded_columns}")
print("\nTarget distribution:")
print(y.value_counts().sort_index())

# ==================================================
# 3. SPLIT DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ==================================================
# 4. DEFINE THREE MODELS
# ==================================================

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}

# ==================================================
# 5. CROSS-VALIDATION AND TESTING
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 65)
    print(f"Training: {name}")
    print("-" * 65)

    # Evaluate generalization on training data using 5-fold CV.
    # F1 for the attack class is the selection metric.
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=1
    )

    # Fit using the complete training split
    model.fit(X_train, y_train)

    # Predict unseen test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # ROC-AUC where probability estimates are available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = model.classes_

        if len(classes) == 2:
            positive_index = list(classes).index(
                sorted(classes)[-1]
            )
            auc = roc_auc_score(
                y_test,
                probabilities[:, positive_index]
            )
        else:
            auc = np.nan
    else:
        auc = np.nan

    results.append({
        "Model": name,
        "CV F1 Mean": cv_scores.mean(),
        "CV F1 Std": cv_scores.std(),
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "Test ROC-AUC": auc
    })

    trained_models[name] = model

    print(f"CV F1:       {cv_scores.mean():.4f} "
          f"+/- {cv_scores.std():.4f}")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1:        {f1:.4f}")

# ==================================================
# 6. COMPARE MODELS IN THE TERMINAL
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["CV F1 Mean", "Test F1"],
    ascending=False
).reset_index(drop=True)

print("\n\n" + "=" * 100)
print("                     MODEL COMPARISON")
print("=" * 100)

print(results_df.to_string(
    index=False,
    formatters={
        col: (lambda x: f"{x:.4f}")
        for col in results_df.select_dtypes(
            include="number"
        ).columns
    }
))

# ==================================================
# 7. SELECT AND SAVE THE BEST MODEL
# ==================================================

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "best_cyber_model.joblib")

# Save the comparison results too
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print(f"BEST MODEL: {best_name}")
print("=" * 65)

print("\nFinal classification report:")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    best_model.predict(X_test)
))

print("\nSaved files:")
print("  1. best_cyber_model.joblib")
print("  2. model_comparison.csv")

print("\nTraining and comparison completed.")
