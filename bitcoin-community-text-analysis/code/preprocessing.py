import pandas as pd
import re
from konlpy.tag import Okt

DATA_PATH = "../data/sample_data.csv"
OUTPUT_PATH = "../data/processed_data.csv"

df = pd.read_csv(DATA_PATH)

# 제목 + 내용 합치기
df['text'] = df['제목'].fillna('') + " " + df['내용'].fillna('')

# 특수문자 제거
def clean(text):
    return re.sub(r"[^가-힣0-9 ]", "", str(text))

df['clean'] = df['text'].apply(clean)

okt = Okt()

# 명사 추출
df['tokens'] = df['clean'].apply(lambda x: okt.nouns(x))

# 짧은 단어 제거
df['tokens'] = df['tokens'].apply(lambda x: [t for t in x if len(t) > 1])

# document 생성
df['document'] = df['tokens'].apply(lambda x: " ".join(x))

df[['document']].to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print("전처리 완료")