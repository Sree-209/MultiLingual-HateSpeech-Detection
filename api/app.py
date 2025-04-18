from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

from src.preprocessing.text_cleaning import clean_text
from src.utils.profanity_utils import load_profanity_dict, extract_profanity_features
from src.features.embeddings import generate_sbert

app = FastAPI()

# === Load artifacts ===
model = joblib.load("models/comment_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/scaler.pkl")
profanity_dict = load_profanity_dict("data/raw/Hinglish_Profanity_List.csv")

# === Request schema ===
class CommentRequest(BaseModel):
    text: str

# === Route ===
@app.post("/predict/")
def predict_comment(data: CommentRequest):
    raw = data.text
    cleaned = clean_text(raw)
    prof_feats = extract_profanity_features(cleaned, profanity_dict)
    sbert_vec = generate_sbert([cleaned])
    tfidf_vec = tfidf.transform([cleaned]).toarray()
    hand_feats_scaled = scaler.transform([prof_feats])

    final_vec = np.hstack([sbert_vec, tfidf_vec, hand_feats_scaled])
    pred = model.predict(final_vec)[0]
    prob = model.predict_proba(final_vec)[0][1]

    return {
        "input": raw,
        "prediction": "Hate" if pred == 1 else "Non-Hate",
        "confidence": float(prob)
    }
