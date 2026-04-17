'''
MLP(다층 퍼셉트론, Multilayer Perceptron)
    MLP 란 여러 개의 퍼셉트론 뉴런을 여러 층으로 쌓은 다층신경망 구조
    입력층과 출력층 사이에 하나 이상의 은닉층(hiden layer)을 가지고 있는 신경망이다.
    피드포워드(순방향) 인공신경망으로, 층을 쌓고 비선형 활성화 함수를 사용하여 
    복잡한 패턴과 XOR 같은 비선형 문제 해결이 가능합니다.
    인접한 두 층의 뉴런간에는 완전 연결 => fully connected 된다.

중요! ※ MLP라고 하지만 이게 Deep Learning의 옛이름
미분이 MLP에서 어떻게 쓰이는가?
    미분으로 오차를 줄여나감

MLP 구조
    입력 -> 신경망(뉴런) -> 출렬 후 오차 확인
    ex) 입력(x) -> 모델 -> 예측값(y^)-실제값(y) -> 오차(loss) 발생

오차함수(Loss Function)
    L = (y - y^)
    예측이 틀릴수록 값이 커짐.

그럼 미분은 왜 쓰나?
    오차를 어떻게 줄일지 즉, 오차가 줄어드는 방향으로 w(weight, 가중치)를 갱신

⁂전체 학습 과정을 보면
    1. 모델이 예측
    2. 오차 계산
    3. 미분(기울기 계산) - 미분계수: 순간변화율
    4. 가중치 W를 갱신
    5. (1. ~ 4.)반복 - 역전파(Back Propergation)
이것이 MLP 학습 -> Deep Learning

MLP에서 max_iter의 추천횟수는 500 ~ 1000
'''
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# 논리 회로
feature = np.array([[0,0],[0,1],[1,0],[1,1]])
and_label = np.array([0, 0, 0, 1]) # AND
or_label = np.array([0, 1, 1, 1]) # OR
xor_label = np.array([1, 0, 0, 1]) # XOR
print(feature)
print()

# AND
print('-'*20,'AND','-'*20)
and_ml2 = MLPClassifier(max_iter=10, hidden_layer_sizes=10, 
                        solver='adam',  # cost 최소화 방식(옵티마이저)
                        learning_rate_init=0.01
                        ).fit(feature, and_label) 
print(and_ml2)  # Perceptron(max_iter=10)
and_pred2 = and_ml2.predict(feature)
print("pred2 :",and_pred2)                            
print("acc :",accuracy_score(and_label, and_pred2))  
print()

# OR
print('-'*20,'OR','-'*20)
or_ml = MLPClassifier(max_iter=10, hidden_layer_sizes=10, 
                        solver='adam',  # cost 최소화 방식(옵티마이저)
                        learning_rate_init=0.01
                        ).fit(feature, or_label)
print(or_ml)  # Perceptron(max_iter=10) 
or_pred = or_ml.predict(feature)
print("pred :",or_pred)                            
print("acc :",accuracy_score(or_label, or_pred))  
print()


# XOR 
print('-'*20,'XOR','-'*20)
xor_ml = MLPClassifier(max_iter=500, hidden_layer_sizes=10, 
                        solver='adam',  # cost 최소화 방식(옵티마이저)
                        learning_rate_init=0.01,
                        verbose=1       # 학습내용 보여줘
                        ).fit(feature, xor_label)
print(xor_ml)  # Perceptron(max_iter=100) 
xor_pred = xor_ml.predict(feature)
print("pred :",xor_pred)                            
print("acc :",accuracy_score(xor_label, xor_pred))  
print()

# 실습 2 - 일반 자료로 분류
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

x, y = make_moons(n_samples=300, noise=0.2, random_state=42)
print(x[:2]) # [[ 0.80392642 -0.29140734]
print(y[:2]) # [1 1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    solver='adam', # default, 옵티마이저
    max_iter=1000,
    random_state=42,
    activation='relu' # default
)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"acc : {accuracy_score(y_test, pred)}") # 0.9666