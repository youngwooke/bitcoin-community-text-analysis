import pandas as pd
from transformers import pipeline
from tqdm import tqdm

DATA_PATH = "../data/sample_data.csv"

df = pd.read_csv(DATA_PATH)

model = pipeline(
    "text-classification",
    model="dudcjs2779/sentiment-analysis-with-klue-bert-base"
)

results = []

for text in tqdm(df['document']):
    result = model(text[:512])[0]
    results.append(result['label'])

df['sentiment'] = results

print(df['sentiment'].value_counts())