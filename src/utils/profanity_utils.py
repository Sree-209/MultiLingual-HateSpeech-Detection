import pandas as pd
from rapidfuzz import fuzz
import chardet

# --------------------------------------
# Load profanity dictionary
# --------------------------------------
def load_profanity_dict(path: str) -> dict:
    with open(path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    encoding = result['encoding']

    profanity_df = pd.read_csv(
        path,
        encoding=encoding,
        names=["Code mixed words", "English Meaning", "Severity scoring"],
        header=None
    )

    profanity_df["Code mixed words"] = profanity_df["Code mixed words"].astype(str).str.strip().str.lower()

    return dict(zip(profanity_df["Code mixed words"], profanity_df["Severity scoring"]))


# --------------------------------------
# Normalize characters for fuzzy matching
# --------------------------------------
def normalize_word(word: str) -> str:
    substitutions = {
        '@': 'a', '0': 'o', '1': 'i', '$': 's', '!': 'i',
        '*': '', '#': '', '%': '', '&': 'and'
    }
    word = word.lower()
    for old, new in substitutions.items():
        word = word.replace(old, new)
    return word


def build_normalized_profanity_dict(profanity_dict_raw: dict) -> dict:
    return {
        normalize_word(word): score for word, score in profanity_dict_raw.items()
    }


# --------------------------------------
# Person-target context checker
# --------------------------------------
def has_person_target(words, index):
    person_targets = {
        "tu", "tum", "tera", "teri", "tujhe", "tujhko", "tumhara", "apna", "apni", "aap", "tumlog", "tumhare",
        "uska", "uski", "uske", "wo", "woh", "unka", "unki", "unke", "unlog", "inlog", "kisi", "kisiko",
        "muslim", "musalman", "hindu", "sikh", "christian", "jain", "maulvi", "pandit", "padri", "mullah", "kafir",
        "pakistani", "indian", "desi", "bihari", "gujju", "madrasi", "bangladeshi", "kashmiri",
        "dalit", "chamar", "bhangi", "brahmin", "sc", "st", "obc", "savarna",
        "ladka", "ladki", "aadmi", "aurat", "banda", "behen", "bhai", "bhen", "launda", "laundi"
    }

    surrounding_window = 2
    context_window = words[max(0, index - surrounding_window):min(len(words), index + surrounding_window + 1)]
    return any(word in person_targets for word in context_window)


# --------------------------------------
# Main feature extractor (person-target only)
# --------------------------------------
def extract_profanity_features(
    text: str,
    profanity_dict: dict,
    max_len: int = 100,
    fuzzy_threshold: int = 85,
    min_len: int = 3,
    max_ngram: int = 4
) -> tuple:
    words = text.split()
    count = 0
    severity_score = 0
    matched_phrases = set()
    norm_profanity_dict = build_normalized_profanity_dict(profanity_dict)

    for n in range(1, max_ngram + 1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i + n]
            ngram = " ".join(ngram_words)
            ngram_clean = normalize_word(ngram.replace(" ", ""))

            if len(ngram_clean) < min_len or ngram_clean in matched_phrases:
                continue

            for profane_word in norm_profanity_dict:
                if fuzz.token_set_ratio(ngram_clean, profane_word) >= fuzzy_threshold:
                    if has_person_target(words, i):
                        count += 1
                        severity_score += norm_profanity_dict[profane_word]
                        matched_phrases.add(ngram_clean)
                    # Else: skip profane word because no personal target
                    break

    binary_match = int(count > 0)
    normalized_len = len(words) / max_len

    return count, severity_score, binary_match, normalized_len
