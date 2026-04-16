'''ex45
Naive Bayes Algorithm을 이용한 분류 - MultinomialNB(다항 나이브 베이즈)
        Spam mail자료를 파일로 받아 분류하기
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import koreanize_matplotlib

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/mydata.csv")
print(df.head(3))
# label값 공백을 없애고, 대문자->소문자로 변경
df['label'] = df['label'].str.strip().str.lower() 
texts = df['text'].tolist() # 2차원 만들기
labels = df['label'].tolist()
print(texts[:3])
print(labels[:3])

# train test spilt
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels,  test_size=0.25, random_state=42, stratify=labels
    )

# 벡터화 시키기
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train) # 단어 사전을 만듦
x_test_vec = vectorizer.transform(x_test)   # test는 transform만 하고 fit하지 않는다.
# print(x_train_vec[:3])

# 모델 생성후 학스
model = MultinomialNB()
model.fit(x_train_vec, y_train)

# 예측하기
y_pred = model.predict(x_test_vec)

# 정확도 확인하기
acc = accuracy_score(y_test, y_pred)
print("분류 정확도 :",acc) # 0.8


# confusion_matrix(혼돈행렬 확인하기)
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print(cm)
# [[2 1]
#  [0 2]]

# ConfusionMatrixDisplay(혼돈행렬 시각화) - heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix(혼돈(혼동)행렬)")
plt.show()

# 사용자가 입력 메일 내용 분류
while True:
    userInput = input("이메일 내용 입력(종료는 : q) : ")
    if userInput.lower() == 'q':
        break
    x_new = vectorizer.transform([userInput])
    prob = model.predict_proba(x_new)[0]
    spam_prob = prob[model.classes_.tolist().index('spam')]

    result = '스펨입니다' if spam_prob >= 0.7 else "정상 메일입니다."
    print(f"스팸 확률은 : {spam_prob:.2f} => {result}")