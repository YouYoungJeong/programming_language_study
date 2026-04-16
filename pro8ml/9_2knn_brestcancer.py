'''ex48
KNN(최근접 이웃_K-Nearest Neighbors)
    brest_cancer 데이터셋 사용
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
x = data.data       # feature
y = data.target     # label  [0:악성(M) 1:양성(B)]
print(x[:2], x.shape)       # (569, 30)
print(y[:2], np.unique(y))  # [0 1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42) # stratify=y : 레이블에 대해서 균형있게

# 스케일링 필요( 거리 기반 모델이므로 크기가 영향을 미침)
# 표준화(StandardScaling) 하기
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) # 학습용은 fit 을 넣어야함
x_test_scaled = scaler.transform(x_test)        # 검증용은 fit없이

# K-NN은 k값이 중요
# k값 변화에 따른 정확도 비교로 최적의 k값 얻기
# k값이 작으면 과적합, k값이 크면 과소적합
train_acc = []
test_acc = []
k_range = range(3, 12)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_scaled, y_train)

    # 예측
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    # 정확도
    train_acc.append(accuracy_score(y_train, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))

# 시각화하기 (보통 시각화해서 많이 찾는다)
plt.figure()
plt.plot(k_range, train_acc, marker='o', label='Train acc')
plt.plot(k_range, test_acc, marker='s', label='Test acc')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.title("K-NN acc comp")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
[그래프 해석]
    그래프 기준으로 최적은 k=3
    test acc가 가장 높은 지점이 3 
    -> train과 test사이의 값의 차이가 있기 때문에 과적합의 위험(의심)이 있고
    4는 불안 급격히 떨어지기 때문에 부적절 하고
    7 ~ 9는 안정적(실무에서는 바람직) train - test차이가 가장 적은 지점이 좋다
    따라서 train - test가 만나는 접점이 있는 k=9를 택하는게 좋다
'''
best_k = k_range[np.argmax(test_acc)]
print('최적의 k : ', best_k) # 3 

# 최종 모델 작성
best_k = 9
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(x_train_scaled, y_train)

# 성능 확인하기
y_pred = final_model.predict(x_test_scaled) 
print(f"final_model 정확도 : {accuracy_score(y_test, y_pred)}") # 0.97368
print(f"분류 리포트\n{classification_report(y_test, y_pred)}")
print(f"confusion_matrix\n{confusion_matrix(y_test, y_pred)}")
# [[39  3]
#  [ 0 72]]

# 새로운 자료로 예측 (그냥 기존 자료를 살짝 수정해서 사용)
new_data = x[0].copy()
new_data = new_data + np.random.normal(0, 0.1, size=new_data.shape)
new_data_scaled = scaler.transform([new_data]) # 스케일링
prediction = final_model.predict(new_data_scaled)
proba = final_model.predict_proba(new_data_scaled)
print("새로운 데이터 예측 결과")
print('예측 :',prediction[0], "/ 참고 : [0:악성(M) 1:양성(B)]")
print('확률 값 :',proba)