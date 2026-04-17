'''ex51
인공신경망(Artificial Neural Network) - Perceptron
Perceptron
    sklearn이 제공하는 단층신경망(뉴런, 노드)
    Perceptron은 딥러닝의 경사하강법과는 달리 틀린것만 고치는 알고리즘이다.
    이항 분류가 가능.
    선형회귀식을 사용(LogisticRegression을 기반으로 함)
        input에 대한 가중치(w)합 계산 후 실제값과 예측값 비교(Loss Function:손실함수)후
        역전파를 통해 W를 갱신을 max_lter만큼 반복함 - 최소제곱법을 사용
    흐름
        예측 후
        => 맞았는지 확인 
        => 틀리면 Weight를 갱신/ 맞으면 통과 
        => 이과정 반복(max_lter만큼 반복 ,딥러닝은 epoch)
'''
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 실습1 - 논리회로 분류
from sklearn.linear_model import Perceptron

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
and_label = np.array([0, 0, 0, 1]) # AND
or_label = np.array([0, 1, 1, 1]) # OR
xor_label = np.array([1, 0, 0, 1]) # XOR

# AND
# max_iter=1
and_ml = Perceptron(max_iter=1).fit(feature,and_label) # max_iter(학습횟수 ,epoch)
print(and_ml)  # Perceptron(max_iter=1) -> 현재 학습을 1번만 해서 역전파를 못함
and_pred = and_ml.predict(feature)
print("pred :",and_pred)                            # [0 0 0 0]
print("acc :",accuracy_score(and_label, and_pred))  # 0.75

# max_iter=10
and_ml2 = Perceptron(max_iter=10).fit(feature, and_label) 
print(and_ml2)  # Perceptron(max_iter=10)
and_pred2 = and_ml2.predict(feature)
print("pred2 :",and_pred2)                            # [0 0 0 1]
print("acc :",accuracy_score(and_label, and_pred2))  # 1.0

# OR
or_ml = Perceptron(max_iter=10).fit(feature, or_label) # max_iter(학습횟수 ,epoch)
print(or_ml)  # Perceptron(max_iter=10) 
or_pred = or_ml.predict(feature)
print("pred :",or_pred)                            
print("acc :",accuracy_score(or_label, or_pred))  


# XOR - 선형 모델이기 때문에 해결을 못한다
xor_ml = Perceptron(max_iter=100).fit(feature, xor_label) # max_iter(학습횟수 ,epoch)
print(xor_ml)  # Perceptron(max_iter=100) 
xor_pred = xor_ml.predict(feature)
print("pred :",xor_pred)                            
print("acc :",accuracy_score(xor_label, xor_pred))  
print()


# 실습2 - 일반 자료 분류
x = np.array([
    [2, 3],
    [3, 3],
    [1, 1],
    [5, 2],
    [6, 1]
])
y = np.array([1, 1, 1, -1, -1])
model = Perceptron(max_iter=100,
                    eta0=0.1,   # 학습률(learning rate)
                    random_state=42)
model.fit(x, y)
pred = model.predict(x)
print('예측값 :', pred)
print('실제값 :', y)
print('정확도 :', accuracy_score(y, pred))

# parameter확인
print(f"가중치(w) : {model.coef_}")
print(f"절편(b, 편향값) : {model.intercept_}")

# 결정 경계(w1*x1 + w2*x2 + b)에 대한 시각화
import matplotlib.pyplot as plt
import koreanize_matplotlib

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')
w = model.coef_[0]
b= model.intercept_[0]
x_vals = np.linspace(0, 7, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals)
plt.title("sklearn_Perceptron Decision Boundary(결정 경계)")
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()