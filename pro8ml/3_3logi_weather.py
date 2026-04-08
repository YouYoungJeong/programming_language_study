'''
Logistic Linear Regression(로지스틱회귀분석)
데이터의 포용성
    train test split 사용 
        모델의 성능을 객관적으로 파악할 수 있다
        모델 학습과 검증에 사용된 자료가 같다면 overfitting(과적합)의 우려발생을 방지하기 위해

날씨 예보 (비가 올지 여부)
    RainTomorrow 범주형 종속변수(label, class)
    나머지열이 독립변수(feature)
dummy화 시킨다 
    범주형 데이터(질적 데이터)를 0과 1의 값을 가지는 숫자 형태(이진 변수)로 변환하는 것
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # 모델 샘플링 추출 모듈
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(data.head(2))
print(data.shape) # (366, 12)
print()

# 데이터 정제하기 
print('-'*100)
data2 = pd.DataFrame()
data2 = data.drop(['Date', 'RainToday'], axis=1)
# 'RainTomorrow'column dummy화 하기
data2['RainTomorrow'] = data2['RainTomorrow'].map({"Yes":1, "No":0})
print(data2.head())
print(data2.shape) # (366, 10)
print(data2.RainTomorrow.unique()) # [1 0]
print()

# 데이터 분리하기
print('-'*100)
print('데이터 분리 : 학습용(train data), 검증용(test data)')
# test size = 0.3-> traindata = 0.7, random_state는 seed
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(train.shape, test.shape) # (256, 10) (110, 10)
print(train.head(3))
print(test.head(3))
print()

# 분류모델 생성 - 학습은 train으로 하기
# 독립변수가 많은 경우 join으로 묶기
print('-'*100)
col_select = "+".join(train.columns.difference(['RainTomorrow']))
print(col_select)
my_formula = 'RainTomorrow ~' + col_select
# model = smf.glm(formula=my_formula, data=train, family=sm.families.Binomial()).fit()
model = smf.logit(formula=my_formula, data=train).fit()
print(model.summary()) 
# P>|z| 0.05보다 큰걸 전부 뜯어내는게 아니라 아주 큰것부터 하나씩 제거하다보면 0.05에 근사해짐.
print(model.params) # coef(b,w)만 보기
print()

# 예측값 확인하기 - 검정운 test로 확인하기
print('예측값 :',np.rint(model.predict(test)[:5].values))
print('실제값 :',test['RainTomorrow'][:5].values)

# 정확도 확인하기
conf_mat = model.pred_table()
print(conf_mat)
# [[197.   9.]
#  [ 21.  26.]]
print("분류 정확도(conf_mat(train)) :",(conf_mat[0][0]+conf_mat[1][1]) / len(train)) # 0.8710
pred = model.predict(test)
print("분류 정확도(accuracy_score(test)) :",accuracy_score(test['RainTomorrow'], np.rint(pred))) # 0.8727
