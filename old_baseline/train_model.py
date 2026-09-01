import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ==============================
# SETTINGS
# ==============================

DATASET_FILE = "features.csv"

MODEL_FOLDER = "models"

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ==============================
# LOAD DATASET
# ==============================

print("\n================================")
print("LOADING DATASET")
print("================================\n")

df = pd.read_csv(DATASET_FILE)

print("Total samples:", len(df))

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .rename({
        0: "REAL",
        1: "FAKE"
    })
)


# ==============================
# PREPARE FEATURES
# ==============================

# Remove non-feature columns

X = df.drop(
    columns=[
        "label",
        "filename"
    ],
    errors="ignore"
)

y = df["label"]


# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Replace missing values

X = X.fillna(
    X.median()
)


print("\nNumber of features:", X.shape[1])


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ==============================
# MODELS
# ==============================

models = {

    "Random Forest":
        Pipeline([
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]),

    "SVM":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

    "Logistic Regression":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])
}


# ==============================
# CROSS VALIDATION
# ==============================

print("\n================================")
print("CROSS-VALIDATION")
print("================================\n")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = {}


for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    results[name] = scores.mean()

    print(
        f"{name}: "
        f"Mean F1 = {scores.mean():.3f}"
    )

    print(
        f"Individual folds: "
        f"{np.round(scores, 3)}\n"
    )


# ==============================
# SELECT BEST MODEL
# ==============================

best_model_name = max(
    results,
    key=results.get
)

print("================================")
print("BEST MODEL")
print("================================")

print(
    f"\nSelected model: {best_model_name}"
)

print(
    f"Cross-validation F1: "
    f"{results[best_model_name]:.3f}"
)


best_model = models[
    best_model_name
]


# ==============================
# TRAIN BEST MODEL
# ==============================

print("\nTraining final model...")

best_model.fit(
    X_train,
    y_train
)


# ==============================
# TEST MODEL
# ==============================

y_pred = best_model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


# ==============================
# RESULTS
# ==============================

print("\n================================")
print("FINAL MODEL RESULTS")
print("================================")

print(
    f"\nModel: {best_model_name}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"F1 Score: {f1:.3f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "REAL",
            "FAKE"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==============================
# SAVE MODEL
# ==============================

model_path = os.path.join(
    MODEL_FOLDER,
    "voice_detector.pkl"
)

joblib.dump(
    best_model,
    model_path
)


print("\n================================")
print("MODEL SAVED")
print("================================")

print(
    f"\nModel: {model_path}"
)

print("\nTraining completed successfully!")