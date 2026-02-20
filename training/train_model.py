import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # 1) Paths
    data_path = os.path.join(os.path.dirname(__file__), "bank_churn_modelling.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Make sure bank_churn_modelling.csv is inside training/."
        )

    model_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend", "model.pkl")
    )

    # 2) Load
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()  # just in case

    # 3) Target
    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"'{target}' column not found. Columns: {df.columns.tolist()}")

    # 4) Drop ID columns if present
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 5) X, y
    X = df.drop(columns=[target]).copy()
    y = df[target]

    # 6) Explicit categorical columns
    categorical_cols = [c for c in ["Geography", "Gender"] if c in X.columns]
    for c in categorical_cols:
        X[c] = X[c].astype(str).str.strip()

    # 7) Numeric columns are everything else
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # 8) Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # 9) Model pipeline
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    # 10) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 11) Train
    clf.fit(X_train, y_train)

    # 12) Evaluate
    preds = clf.predict(X_test)
    print("\n=== Model Evaluation ===")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # 13) Save
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump(clf, f)

    print("\nModel saved to:", model_out)


if __name__ == "__main__":
    main()
