# 비트코인 커뮤니티 텍스트 분석

비트코인 관련 온라인 커뮤니티 게시글을 대상으로 형태소 분석, 임베딩 기반 유사도 분석, 군집 분석, 감정 분석을 수행하여 투자 정보의 신뢰성을 평가하는 프로젝트입니다.

## 분석 개요

- 수집 대상: DCInside 비트코인 갤러리 게시글
- 수집 도구: Octoparse
- 분석 규모: 보고서 기준 게시글 4,301건
- 주요 방법: Okt 형태소 분석, Sentence-BERT 임베딩, 코사인 유사도, KMeans 군집 분석, KLUE-BERT 기반 감정 분석

## 실행 순서

아래 명령은 `code` 폴더에서 실행합니다.

```bash
python preprocessing.py
python embedding.py
python clustering.py
python sentiment.py
```

## 스크립트 설명

- `preprocessing.py`: 제목, 본문, 댓글을 결합하고 특수문자 정제 후 Okt로 명사를 추출하여 `data/processed_data.csv`를 생성합니다.
- `embedding.py`: Sentence-BERT로 문서 임베딩을 생성하고, 핵심 투자 용어 쌍의 코사인 유사도를 `results/core_word_similarity.csv`로 저장합니다.
- `clustering.py`: 문서 임베딩을 기반으로 KMeans 군집 분석을 수행하고 `results/cluster_results.csv`를 저장합니다.
- `sentiment.py`: 원문 텍스트를 대상으로 감정 분석을 수행하고 `results/sentiment_results.csv`, `results/sentiment_summary.csv`를 저장합니다.

## 프로젝트 구조

```text
bitcoin-community-text-analysis/
├── code/
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── clustering.py
│   └── sentiment.py
├── data/
│   └── sample_data.csv
├── results/
│   ├── noun_frequency.png
│   └── sentiment_distribution.png
└── README.md
```

## 데이터 안내

원본 데이터는 저작권 및 개인정보 이슈로 공개하지 않고, 저장소에는 일부 샘플 데이터만 포함합니다.

## 재현성 메모

감정 분석과 임베딩 모델은 Hugging Face 모델을 내려받아 실행하므로 최초 실행 시 네트워크 연결이 필요합니다. Okt 형태소 분석을 사용하려면 Java와 `JAVA_HOME` 설정이 필요합니다. Java 환경이 없으면 샘플 재현을 위해 정규식 기반 토큰 추출로 대체 실행됩니다. 보고서의 수치는 전체 데이터 기준 결과이며, 저장소의 샘플 데이터만 실행할 경우 결과가 다를 수 있습니다.
