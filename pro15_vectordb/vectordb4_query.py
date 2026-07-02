# CromaDB 에 ...

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path='.aa/bb/ccdb')
collection = client.get_or_create_collection(name='mytest', metadata={'hnsw:space':'l2'})
# metadata={'hnsw:space':'l2'} : 벡터 데이터베이스에서 고차원 벡터 간의 유사도를 측정할 때 L2 Norm (유클리드 거리)을 사용하도록 설정하는 코드
# l2 외에도 코사인 유사도를 뜻하는 cosine, 내적을 뜻하는 ip로 변경 가능

texts = [
    'Apple is a fruit',
    'Python is a programming language',
    'The sun rise in the east',
    'I love to eat mangoes'
]

ids = [str(i) for i in range(len(texts))]
print(ids)


model = SentenceTransformer('all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension()) # 384
print(model)
# SentenceTransformer(
#   (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
#   (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
#   (2): Normalize({})
# )
print('\n')

embeddings = model.encode(texts).tolist()
# print(embeddings)
print(model.encode(texts).shape)    # (4, 384)
print('\n')


# vectordb 에 저장
collection.add(
    documents=texts,
    ids=ids,
    embeddings=embeddings
)


# vectordb 자료 조회
record = collection.get(ids=['0'], include=['embeddings', 'documents'])
print('조회된 문서 : ', record['documents'][0])
print('벡터(앞 10개) : ', record['embeddings'][0][:10])



#######################################################################################
# vectordb 의 자료 유사 문장 검색
query_data = "What is Python?"    # DB에 없는 문장
query_vector = model.encode([query_data]).tolist()
# print(query_vector)     # 벡터화 완료

result = collection.query(
    query_embeddings=query_vector,
    n_results=2,
    include=['documents', 'distances']      # distances - query vector 와의 거리 → 0에 근사하면 유사함
)

print('유사한 문장 겸색 결과 : ')
for doc, dist in zip(result['documents'][0], result['distances'][0]):
    print(f'- 문장 : {doc} (유사도 거리 : {dist:.4f})')