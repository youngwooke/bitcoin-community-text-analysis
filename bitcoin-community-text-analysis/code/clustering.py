from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
N_CLUSTERS = 5


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[df["document"].fillna("").astype(str).str.len() > 0].copy()

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["document"].astype(str).tolist(),
        show_progress_bar=True,
    )

    n_clusters = min(N_CLUSTERS, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "cluster_results.csv"
    df[["document", "cluster"]].to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = df["cluster"].value_counts().sort_index()
    print(counts)
    print(f"Cluster results saved: {output_path}")


if __name__ == "__main__":
    main()
