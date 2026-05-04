from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed_data.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "dudcjs2779/sentiment-analysis-with-klue-bert-base"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    texts = df["raw_text"].fillna(df["document"]).astype(str)

    classifier = pipeline("text-classification", model=MODEL_NAME)

    labels = []
    scores = []
    for text in tqdm(texts, desc="Sentiment analysis"):
        result = classifier(text[:512])[0]
        labels.append(result["label"])
        scores.append(result.get("score"))

    df["sentiment"] = labels
    df["sentiment_score"] = scores

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "sentiment_results.csv"
    df[["raw_text", "sentiment", "sentiment_score"]].to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )

    counts = df["sentiment"].value_counts()
    ratios = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percent": ratios})
    summary_path = RESULTS_DIR / "sentiment_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print(summary)
    print(f"Sentiment results saved: {output_path}")
    print(f"Sentiment summary saved: {summary_path}")


if __name__ == "__main__":
    main()
