# Embedding Vector 시각화 및 chromaDB 에 저장 후 검색

# pip install plotly
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px     # 3D 산점도 그래프 작성용 라이브러리
import numpy as np
import pandas as pd


# Data
texts = [  # 임베딩할 원본 문장 목록 (4개 주제, 각 5개 문장)
    "김치찌개는 김치와 돼지고기를 넣고 끓이는 한국의 대표적인 찌개이다.",
    "된장찌개는 된장을 기본 재료로 하여 끓이는 한국 전통 음식이다.",
    "비빔밥은 밥 위에 여러 가지 나물과 고추장을 넣어 비벼 먹는 음식이다.",
    "불고기는 양념한 소고기를 구워 먹는 한국의 대표적인 고기 요리이다.",
    "떡볶이는 매운 고추장 양념에 떡을 넣고 조리하는 간식이다.",
    "파이썬은 문법이 간결하여 초보자가 배우기 쉬운 프로그래밍 언어이다.",
    "자바는 기업용 백엔드 시스템 개발에 많이 사용되는 객체지향 언어이다.",
    "자바스크립트는 웹 브라우저에서 동작하는 대표적인 프로그래밍 언어이다.",
    "스프링 부트는 자바 기반의 웹 애플리케이션 개발을 쉽게 도와주는 프레임워크이다.",
    "SQL은 데이터베이스에서 데이터를 조회하고 관리하기 위한 언어이다.",
    "클라우드는 인터넷을 통해 서버와 저장소 같은 컴퓨팅 자원을 제공하는 기술이다.",
    "AWS는 전 세계적으로 많이 사용되는 대표적인 클라우드 서비스 플랫폼이다.",
    "가상 머신은 하나의 물리 서버 위에서 여러 운영체제를 실행할 수 있게 해준다.",
    "컨테이너는 애플리케이션 실행 환경을 가볍게 패키징하는 기술이다.",
    "쿠버네티스는 컨테이너를 자동으로 배포하고 관리하는 오케스트레이션 도구이다.",
    "인공지능은 데이터를 학습하여 판단하거나 예측하는 기술이다.",
    "머신러닝은 데이터에서 패턴을 찾아 새로운 데이터에 대한 예측을 수행한다.",
    "딥러닝은 인공신경망을 이용하여 복잡한 문제를 학습하는 머신러닝 방법이다.",
    "자연어 처리는 컴퓨터가 사람의 언어를 이해하고 생성하도록 만드는 기술이다.",
    "컴퓨터 비전은 이미지와 영상을 분석하여 객체를 인식하는 인공지능 분야이다."
]


# 각 문장 주제 라벨 작성
categories = [
    "음식","음식","음식","음식","음식",
    "프로그래밍","프로그래밍","프로그래밍","프로그래밍","프로그래밍",
    "클라우드","클라우드","클라우드","클라우드","클라우드",
    "인공지능","인공지능","인공지능","인공지능","인공지능"
]


# 검색에 사용할 질문 
query = "파이썬은 인공지능 개발에 사용되나요?"

all_texts = texts + [query]
all_categories = categories + ["질문"]

labels = [f"문장{i + 1}" for i in range(len(texts))] + ['질문']
print(labels)   # ['문장1', '문장2', '문장3', '문장4', ... '질문']


# Embedding Model 사용
model = SentenceTransformer('all-MiniLM-L6-v2')


# 문장과 질문을 Embedding vector 로 변환
embeddings = model.encode(
    all_texts,
    normalize_embeddings=True   # 벡터 길이를 1로 정규화하여, 코사인 유사도 계산에 적합하도록 함
)
print('Embedding vector 차원 : ', embeddings.shape)     # (21, 384)


# Embedding 된 문장과 질문을 분리 
doc_embeddings = embeddings[:-1]
query_embeddings = embeddings[-1]


# 질문 Vector와 각 문장의 코사인 유사도 확인
similarities = np.dot(doc_embeddings, query_embeddings)     # 내적

top_k = 3
top_indices = similarities.argsort()[::-1][:top_k]

print('\n질문 : ', query)
print('질문과 유사한 문장 Top3 : ')
for rank, idx in enumerate(top_indices, start=1):
    print(f'{rank}) 유사도 : {similarities[idx]:.4f}')
    print(f'   주제 : {categories[idx]}')
    print(f'   문장 : {texts[idx]}')


#######################################################################################
# Embedding Vector Visualization (384 → 3 dimension)
pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)
print('축소 후 shape : ', reduced.shape)    # (21, 3)


# Dataframe for visualization
df = pd.DataFrame({
    '라벨':labels,
    '문장':all_texts,
    '주제':all_categories,
    'x':reduced[:, 0],
    'y':reduced[:, 1],
    'z':reduced[:, 2]
})
# print(df)


# Visualization - Scatter
fig = px.scatter_3d(
    data_frame=df,
    x='x',
    y='y',
    z='z',
    color='주제',
    text='라벨',
    hover_name='문장',
    hover_data={
        '주제':True,
        'x':True,
        'y':True,
        'z':True
    },
    title='문장 Embedding Vector 3D 시각화 + 질문 문장 비교'
)

# 점 크기와 투명도 조정
fig.update_traces(
    marker=dict(size=7, opacity=0.8),
    textposition='top center'
)

# 그래프 축 제목
fig.update_layout(
    scene=dict(xaxis_title='PCA1', yaxis_title='PCA2', zaxis_title='PCA3')
)

import plotly.io as pio
pio.renderers.default = "browser"
# fig.show()

# HTML 로 저장
# fig.write_html('Embedding_vector_3D.html', auto_open=True)



#######################################################################################
# ChromaDB 에 저장 후 질문에 유사한 문장 검색
import os 
from chromadb import PersistentClient
import shutil

if os.path.exists('.chroma_demo'):
    shutil.rmtree('.chroma_demo')

chroma_client = PersistentClient(path='.chroma_demo')

collection = chroma_client.get_or_create_collection(
    name='my_docs',
    metadata={'hnsw:space':'cosine'}    # 코사인 거리 기준 알고리즘 사용
)

ids = [f'doc_{i}' for i in range(len(texts))]
print(ids)

metadatas = [
    {
        'category':categories[i],
        'label':f'문장{i + 1}'
    }
    for i in range(len(texts))
]
# print(metadatas)

collection.add(
    ids=ids,
    documents=texts,
    embeddings=doc_embeddings.tolist(),
    metadatas=metadatas
)
print('저장된 문장 수 : ', collection.count())



#######################################################################################
# 질문과 유사한 문장 검색
results = collection.query(
    query_embeddings=[query_embeddings.tolist()],
    n_results=3,
    include=['documents', 'metadatas', 'distances']
)
# print(results)
print('검색 질문 : ', query)
print('검색 결과 Top 3 :')
result_ids = results['ids'][0]
results_docs = results['documents'][0]
results_metas = results['metadatas'][0]
results_dist = results['distances'][0]

for rank, (doc_id, doc, meta, dist) in enumerate(
    zip(result_ids, results_docs, results_metas, results_dist), 
    start=1
):
    print(f'\n[{rank}]')
    print(f'ID : {doc_id}')
    print(f'라벨 : {meta["label"]}')
    print(f'주제 : {meta["category"]}')
    print(f'거리 : {dist:.4f}')
    print(f'문장 : {doc}')    print(f'라벨 : {meta["label"]}')