'''
단순선형회귀 - iris dataset사용
상관관계가 약한경우와 강한경우로 분석모델을 생성 후 비교
OLS Regression Results
'''
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris.head())
print(type(iris)) # <class 'pandas.core.frame.DataFrame'>
print()

#상관관계 확인하기
print('-----corr확인하기------')
print(iris.iloc[:, 0:4].corr())

'''=============================================================================='''
print("연습1 - 상관관계 약한 변수 사용 : -0.117570 ")
# 모델생성
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()

# 요약정보 보기
# print(result1.summary())
# 1st) F-statistic:2.074 -> 0.05보다 크네 의미가 없다.

print("R-squared : ", result1.rsquared) # 0.0138226 < 0.05보다 작음 의미가 없다.
print('p-values :', result1.pvalues) # 1.518983e-01 > 0.05 이 모델은 유의하지 않다.
print()

# 시각화
# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color='r')
# plt.show()

'''=============================================================================='''
print("연습2 - 상관관계 강한 변수 사용 : 0.871754 ")
# 모델생성
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()

# 요약정보 보기
print(result2.summary())
# 1st) F-statistic:1.04e-47 < 0.05 완전 의미 있다.

print("R-squared : ", result2.rsquared) # 0.760 > 0.05 의미있는 데이터
print('p-values :', result2.pvalues) # 1.038667e-47 < 0.05 이 모델은 유의하다.
print()

# 시각화
# plt.scatter(iris.petal_length, iris.sepal_length)
# plt.plot(iris.petal_length, result2.predict(), color='b')
# plt.show()

# 실제 값으로 예측하기
print("실제값 : ", iris.sepal_length[:5].values)
print("모델이 예측한 값 : ",result2.predict()[:5])
# 실제값 :              [5.1        4.9         4.7         4.6         5. ]
# 모델이 예측한 값 :    [4.8790946  4.8790946  4.83820238 4.91998683 4.8790946 ]
print()

# 새로운 값으로 예측하기
new_data = pd.DataFrame({"petal_length":[1.1, 0.5, 6.0]})
y_pred = result2.predict(new_data)
print("예측 결과 :", y_pred.values) # [4.75641792 4.51106455 6.76013708]
print()

'''=============================================================================='''
print("연습3 - 독립변수를 복수로 사용 (다중선형회귀) ")
# 모델생성 방법1 - 전부 적어서(+)
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width', data=iris).fit()

# 모델생성 방법2 - 제끼는 방법(.join(iris.columns.difference)
column_select = "+".join(iris.columns.difference(['sepal_length','sepal_width','species'])) 
# .difference(독립변수(x)로 사용하지 않을 녀셕들을 적어준다.))
# print(column_select) # petal_length+petal_width
result3 = smf.ols(formula='sepal_length ~'+column_select, data=iris).fit()

# 요약정보 보기
print(result3.summary()) #  Adj. R-squared: 0.763, Prob (F-statistic): 4.00e-47 유의함.