import joblib
import numpy as np
from src.preprocessing.text_cleaning import clean_text
from src.utils.profanity_utils import extract_profanity_features
from src.features.embeddings import sbert

model = joblib.load("models/comment_model.pkl")
scaler = joblib.load("models/scaler.pkl")
tfidf = joblib.load("models/tfidf.pkl")
profanity_dict = ...  # load once at start

def predict(text):
    clean = clean_text(text)
    sb_vec = sbert.encode([clean])
    tfidf_vec = tfidf.transform([clean]).toarray()
    hand_feats = extract_profanity_features(clean, profanity_dict)
    hand_feats_scaled = scaler.transform([hand_feats])
    full = np.hstack([sb_vec, tfidf_vec, hand_feats_scaled])
    return model.predict(full)[0]
