# Embedding 방법 정리
# ChromaDB는 Embedding 모델이 아니라 Embedding 결과를 저장하는 DB
# Embedding이 선행 되어야 함 - 방법이 여러가지...

import chromadb
from sentence_transformers import SentenceTransformer


client = chromadb.PersistentClient(path='.chroma_db')
texts = ['사과는 과일이야', '고양이는 동물이야']


#######################################################################################
# 방법 1 : 가능하나, 비권장
# ChromaDB 내부 Embedding 함수를 직접 꺼내 사용하는 방식
collection1 = client.get_or_create_collection(name='test')
embedding_fn1 = collection1._embedding_function
embeddings1 = embedding_fn1(texts)
print(len(embeddings1), len(embeddings1[0]))    # 문장-2,  차원수-384
print(embeddings1[0][:5])   # [-0.04257268  0.08842303  0.00592434  0.01293706 -0.01990776]
print('\n')


collection1.upsert(     # update + insert
    documents=texts,
    embeddings=embeddings1,
    ids=['id1', 'id2']
)   # 저장 완료, 조회는 생략



#######################################################################################
# 방법 2 : ChromaDB 에서 가장 일반적인 방법 - 추천!
# ChromaDB 에 Embedding 함수를 등록해 자동 Embedding 하는 방식
from chromadb.utils import embedding_functions

embedding_fn2 = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'   # 모델명을 직접 지정
)
embeddings2 = embedding_fn2(texts)

print(len(embeddings2), len(embeddings2[0])) 
print(embeddings2[0][:5])
print('\n')

collection2 = client.get_or_create_collection(name='test2', embedding_function=embedding_fn2)

collection2.upsert(     # update + insert
    documents=texts,
    ids=['id1', 'id2']  # collection2 생성할 때 Embedding 지정했으므로, 안적어도 됨
)   # 저장 완료, 조회는 생략



#######################################################################################
# 방법 3 : ChromaDB와 별개로 SentenceTransformer 모델을 직접 사용해 Embedding을 만드는 방법
# SentenceTransformer로 직접 Embedding

model3 = SentenceTransformer('all-MiniLM-L6-v2')
embeddings3 = model3.encode(texts).tolist()

collection3 = client.get_or_create_collection(name='test3')
collection3.upsert(     # update + insert
    documents=texts,
    embeddings=embeddings3,
    ids=['id1', 'id2']
)   # 저장 완료, 조회는 생략



#######################################################################################
# 방법 4 : Hugging Face 의 사전 학습 모델을 로컬에서 사용하는 방법
# Hugging Face : AI 모델 원격 저장소 - 잘 찾아서 사용하는 것도 좋은 방법

embedding_fn4 = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='jhgan/ko-sroberta-multitask'
)
embeddings4 = embedding_fn4(texts)

print(len(embeddings4), len(embeddings4[0]))    # 2,   768 → 차원이 다름
print(embeddings4[0][:5])
print('\n')

collection4 = client.get_or_create_collection(name='test4', embedding_function=embedding_fn4)
collection4.upsert(     # update + insert
    documents=texts,
    ids=['id1', 'id2']
)   # 저장 완료, 조회는 생략

# 코드보다 어떤 Embedding 함수를 선택하고 이해하는지가 중요함!!