# Chromadb에 add(저장), update(수정), delete(삭제), get(조회) 연습

import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Client, Embedding Function
embedding_fn = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
client = PersistentClient(path='.chroma')

collection = client.get_or_create_collection(name='text', embedding_function=embedding_fn)


#######################################################################################
# Data 저장 - add()
collection.add(
    documents=[
        "문서1 : 인공지능 기술이 난리가 났네",
        "문서2 : 언제다 공부하나"
    ],
    metadatas=[
        {"tag":"mes1"},
        {"tag":"mes2"}
    ],
    ids = [
        "doc1",
        "doc2"
    ]
)


#######################################################################################
# Data 조회 - get()
print('\n전체 문서 조회')
results = collection.get(include=['documents', 'metadatas','embeddings'])   # 'ids'는 기본적으로 읽힘
# print(results)

for doc, meta, emb, id in zip(results['documents'], results['metadatas'], results['embeddings'], results['ids']):
    print(f'id : {id}')
    print(f'documents : {doc}')
    print(f'metadats : {meta}')
    print(f'embeddings : {len(emb)}, {emb[:5]}')
    print('-'*30)



#######################################################################################
# Data 수정 - update()
collection.update(
    ids=['doc2'],
    documents=['문서2 : 내용을 일부 수정'],
    metadatas=[{'tag':'edited message'}]
)

print('\n수정 후 자료 읽기')
upresults = collection.get(where={'tag':'edited message'}, include=['documents', 'metadatas'])

for doc, meta in zip(upresults['documents'], upresults['metadatas']):
    print(f'documents : {doc}')
    print(f'metadats : {meta}')
    print('-'*30)



#######################################################################################
# Data 삭제 - delete()
collection.delete(ids=['doc1'])
# collection.delete(where={'tag':'mes1'})

print('\n삭제 후 문서 조회')
upresults = collection.get(include=['documents', 'metadatas'])

for doc, meta in zip(upresults['documents'], upresults['metadatas']):
    print(f'documents : {doc}')
    print(f'metadats : {meta}')
    print('-'*30)