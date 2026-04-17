'''ex54
MLP(다층 퍼셉트론, Multilayer Perceptron)
    wine등급을 3등급으로 으로 나눔
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_wine()
x = data.data
y = data.target
print(x[:2]," ",x.shape)        # (178, 13)
print(x[:2]," ", np.unique(y))  # [0 1 2]

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 - mlp는 표준화를 권장함
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 모델 생성
model = MLPClassifier(
    hidden_layer_sizes=(20, 10),  # node:50개, node:30개인 은닉층 layer2개
    activation='relu',            # 활성화함수
    solver='adam',                # 손실 최소화 함수 지정
    learning_rate_init= 0.001,    # 학습률
    max_iter= 150,                 # 학습 횟수
    verbose= 1,                   # 학습 도중 로그 출력 여부
    random_state= 42              
)
model.fit(x_train_scaled, y_train)
pred = model.predict(x_test_scaled)
print(f'accuracy_score : {accuracy_score(y_test, pred)}')     # 0.8333
print(f'classification_report : \n{classification_report(y_test, pred)}')


# 혼돈 행렬 시각화(confusion_matrix)
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("혼돈 행렬 시각화(confusion_matrix)")
plt.xlabel("Predicted")
plt.ylabel("실제값(actual)")
plt.show()

# train Loss가 떨어지 과정 시각화 - train loss curve
plt.plot(model.loss_curve_)
plt.title("train loss curve")
plt.xlabel("iteration(epoch)")
plt.ylabel("예측값과 실제값의 차이(Loss)") 
plt.show()