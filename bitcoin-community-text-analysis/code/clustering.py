import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

DATA_PATH = "../data/processed_data.csv"

df = pd.read_csv(DATA_PATH)

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

embeddings = list(df['document'].apply(lambda x: model.encode(x)))

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

print(df['cluster'].value_counts())