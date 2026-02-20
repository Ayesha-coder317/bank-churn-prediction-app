import os
import pickle
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_churn(features: dict) -> int:
    """
    Convert API snake_case fields to original training column names,
    then run prediction.
    """

    # Map API names -> training dataset names
    column_mapping = {
        "Num_Of_Products": "Num Of Products",
        "Has_Credit_Card": "Has Credit Card",
        "Is_Active_Member": "Is Active Member",
        "Estimated_Salary": "Estimated Salary",
    }

    # Rename keys
    for api_key, real_key in column_mapping.items():
        if api_key in features:
            features[real_key] = features.pop(api_key)

    df = pd.DataFrame([features])

    pred = model.predict(df)[0]
    return int(pred)

