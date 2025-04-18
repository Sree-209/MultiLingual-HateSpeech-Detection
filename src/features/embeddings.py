from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def generate_sbert(texts):
    return sbert.encode(texts, show_progress_bar=True)

def generate_tfidf(texts):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    tfidf_features = tfidf.fit_transform(texts).toarray()
    return tfidf, tfidf_features
