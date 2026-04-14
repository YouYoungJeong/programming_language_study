''' ex35
SVM (Support Vector Machine) - SVM으로 XOR(논리회로) 처리하기.
    [논리 회로](XOR 분류시 곡선의 형태가 됨 - logistic의 방식으로 불가하다.SVM은 가능)
        AND     OR      XOR
    00   0      0        0
    01   0      1        1
    10   0      1        1
    11   1      1        0
'''
# AND
and_data = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]
# OR
or_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

# XOR
xor_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

# # data - feature, label 분리 방법 1 - for문 사용
# feature = []
# label = []
# for row in x_data:
#     p = row[0]
#     q = row[1]
#     r = row[2]
#     feature.append([p,q])
#     label.append(r)
# print(feature)
# print(label)

# data - feature, label 분리 방법 2 - pandas사용
and_df = pd.DataFrame(and_data)
and_feature = np.array(and_df.iloc[:, 0:2])
and_label = np.array(and_df.iloc[:,2])
# print(and_feature)
# print(and_label)

lmodel = LogisticRegression() # 선형 분리 모델
smodel = svm.SVC() # 선형 / 비선형 분리 모델

lmodel.fit(and_feature, and_label)
smodel.fit(and_feature, and_label)

# 예측값
lpred = lmodel.predict(and_feature)
print(f'lmodel 예측값 : {lpred}')
spred = smodel.predict(and_feature)
print(f'smodel 예측값 : {spred}')
print()

# 정확도
acc1 = metrics.accuracy_score(and_label, lpred)
print(f'lmodel 분류 정확도 : {acc1}') # 0.75
acc2 = metrics.accuracy_score(and_label, spred)
print(f'smodel 분류 정확도 : {acc2}') # 1.0
print()


# OR =============================================================================
# data - feature, label 분리
or_df = pd.DataFrame(or_data)
or_feature = np.array(or_df.iloc[:, 0:2])
or_label = np.array(or_df.iloc[:,2])
# print(or_feature)
# print(or_label)

lmodel = LogisticRegression() # 선형 분리 모델
smodel = svm.SVC() # 선형 / 비선형 분리 모델

lmodel.fit(or_feature, or_label)
smodel.fit(or_feature, or_label)

# 예측값
lpred = lmodel.predict(or_feature)
print(f'or_lmodel 예측값 : {lpred}')
spred = smodel.predict(or_feature)
print(f'or_smodel 예측값 : {spred}')
print()

# 정확도
acc1 = metrics.accuracy_score(or_label, lpred)
print(f'or_lmodel 분류 정확도 : {acc1}') # 0.75
acc2 = metrics.accuracy_score(or_label, spred)
print(f'or_smodel 분류 정확도 : {acc2}')
print()

# XOR ==========================================================================
# data - feature, label 분리
xor_df = pd.DataFrame(xor_data)
xor_feature = np.array(xor_df.iloc[:, 0:2])
xor_label = np.array(xor_df.iloc[:,2])
# print(xor_feature)
# print(xor_label)

lmodel = LogisticRegression() # 선형 분리 모델
smodel = svm.SVC() # 선형 / 비선형 분리 모델

lmodel.fit(xor_feature, xor_label)
smodel.fit(xor_feature, xor_label)

# 예측값
lpred = lmodel.predict(xor_feature)
print(f'xor_lmodel 예측값 : {lpred}')
spred = smodel.predict(xor_feature)
print(f'xor_smodel 예측값 : {spred}')
print()

# 정확도
acc1 = metrics.accuracy_score(xor_label, lpred)
print(f'xor_lmodel 분류 정확도 : {acc1}') # 0.75
acc2 = metrics.accuracy_score(xor_label, spred)
print(f'xor_smodel 분류 정확도 : {acc2}') # 1.0