'''ex44
Naive Bayes Algorithm을 이용한 분류 - MultinomialNB(다항 나이브 베이즈)
    베이즈 정리를 기반으로 한 머신러닝 분류 알고리즘으로, 
    주로 텍스트 분류나 빈도수 기반 데이터를 처리할 때 사용
    단어의 출현 빈도(count)를 바탕으로 문서를 분류하는 데 탁월함
    형태 : 단어가 많이 나왓으면 그 단어 빈도에 비례해 확률이 커짐
        Spam mail분류하기
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 학습용 데이터
texts = [
    '무료 쿠폰 지금 무료 클릭하면 무료 선물',
    '한번만 클릭하면 무료 무료 대박',
    '오늘 회의는 2시야',
    '지금 할인 행사 진행중',
    '회의 자료는 메일로 보내주세요',
    '지금 바로 쿠폰 확인'
]
label = ['spam','spam','ham','spam','ham','spam']

# 단어 등장 횟수 기반 벡터 만들기
# CountVectorizer(): 문서들로부터 단어의 순서 정보는 버리고, 단어의 빈도수 정보를 벡터로 변환하는 도구
vect = CountVectorizer() 
x = vect.fit_transform(texts)
print(vect.get_feature_names_out()) # 공백을 기준으로 자름
# ['2시야' '메일로' '무료' '바로' '보내주세요' '생하' ...]
print(x) 
#Coords(문서번호, 단어순서) Values(등장횟수) 
# Coords        Values
#   (0, 2)        3    : 0번문서 무료라는 단어 2개
print(x.toarray())
# [[0 0 0 3 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0]... -실제 행렬(희소벡터) 
print(vect.vocabulary_) # index확인하기
# {'무료': 2, '쿠폰': 10, '지금': 8, '클릭': 11, '한번만': 13, '클릭하면': 12,...

# 모델을 생성후 학습
mmodel =MultinomialNB()
mmodel.fit(x, label)

# 정확도 확인하기
from sklearn.metrics import accuracy_score
mpred = mmodel.predict(x)
print('정확도 :',accuracy_score(label, mpred)) # 본인걸로 훈련해서 1.0

# 새로운 문장 테스트
test_text = ['무료 쿠폰 지금 발급','간부 회의는 언제 시작하나요?']
x_test = vect.transform(test_text)
print(x_test)

# 새로운 데이터값 예측 + 확률
new_pred = mmodel.predict(x_test)
probs = mmodel.predict_proba(x_test)
class_names = mmodel.classes_ # ['ham','spam']

for text, pred, prob in zip(test_text, new_pred, probs):
    prob_str = ", ".join([f'{cls}:{p:.4f}' for cls, p in zip(class_names, prob)])
    print(f"'{text} -> 예측{pred} / 확률:[{prob_str}]")