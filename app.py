from fastapi import FastAPI
import joblib, pandas as pd

model = joblib.load("best_cyber_model.joblib")
FEATURES = list(model.feature_names_in_)
app = FastAPI()

@app.post("/predict")
def predict(row: dict):
    X = pd.DataFrame([row]).reindex(columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    return {"is_attack": int(proba >= 0.5), "attack_probability": round(proba, 4)}
