from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

WORD_PAIRS = [
    ("매수", "매도"),
    ("매수", "매입"),
    ("매도", "매매"),
    ("보유", "자산"),
    ("보유", "매입"),
    ("거래소", "코스닥"),
]


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    documents = df["document"].fillna("").astype(str).tolist()

    model = SentenceTransformer(MODEL_NAME)
    document_embeddings = model.encode(documents, show_progress_bar=True)
    print(f"Document embeddings created: {len(document_embeddings)}")

    rows = []
    for word1, word2 in WORD_PAIRS:
        vectors = model.encode([word1, word2])
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        rows.append(
            {
                "word1": word1,
                "word2": word2,
                "cosine_similarity": round(float(score), 3),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "core_word_similarity.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Core word similarity saved: {output_path}")


if __name__ == "__main__":
    main()
