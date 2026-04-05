import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "../data/sample_data.csv"

df = pd.read_csv(DATA_PATH)

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

df['embedding'] = df['document'].apply(lambda x: model.encode(x))

print("임베딩 완료")