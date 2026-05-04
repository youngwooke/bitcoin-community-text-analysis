from pathlib import Path
import re

import pandas as pd

try:
    from konlpy.tag import Okt
except ImportError:
    Okt = None


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sample_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed_data.csv"


def clean_text(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def fallback_nouns(text):
    return re.findall(r"[가-힣a-zA-Z0-9]{2,}", text)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    title = df["제목"].fillna("") if "제목" in df.columns else ""
    body = df["내용"].fillna("") if "내용" in df.columns else ""
    comments = df["댓글"].fillna("") if "댓글" in df.columns else ""

    df["raw_text"] = title + " " + body + " " + comments
    df["clean_text"] = df["raw_text"].apply(clean_text)

    try:
        if Okt is None:
            raise RuntimeError("KoNLPy is not installed.")
        okt = Okt()
        df["tokens"] = df["clean_text"].apply(
            lambda text: [token for token in okt.nouns(text) if len(token) > 1]
        )
        tokenizer_name = "Okt"
    except Exception as exc:
        print(f"Okt is unavailable, using regex fallback tokenizer: {exc}")
        df["tokens"] = df["clean_text"].apply(fallback_nouns)
        tokenizer_name = "regex_fallback"
    df["document"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["raw_text", "clean_text", "tokens", "document"]].to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8-sig"
    )

    print(f"Preprocessing complete ({tokenizer_name}): {len(df)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
