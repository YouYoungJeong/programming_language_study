'''
[선형회귀]
전통적인 방법의 선형회귀(ML중 지도학습에 해당하는) 모델 4가지 알아보기
    각 데이터에 대한 잔차제곱합이 최소가 되는 추세선(회귀선)을 만들고,
    이를 통해 독립변수가 종속변수에 얼마나 영향을 주는지 인과관계를 분석
    독립변수 : 연속형,  종속변수 : 연속형 - 두 변수는 상관관계 및 인과관계가 있어야함.
    정량적인 모델을 생성
    전부 잔차제곱법을 이용하기 때문에 편미분을 사용하고 있다.
'''
import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

np.random.seed(12)

'''=====================================================================
방법 1 : make_regression사용 - model을 만들지는 X
bias 의 default는 0(0,0)이다.
n_features 독립변수
====================================================================='''
print('-'*20,'방법 1','-'*20)
x , y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x,"\n")   # 독립변수(feature) 2차원배열    -[[-1.70073563] [-0.67794537]
print(y,"\n")   # 종속변수(target:y) 1차원 배열  - [-52.17214291 39.34130801 
print(coef)     # 기울기 - 89.47430739278907
print()
# 회귀식 완성됨. -> y = wx + b == y^ = (- 89.47430739278907)*x + 100
y_pred = coef * (-1.70073563) + 100
print("예측값 : ",y_pred) # y = -52.17214291 : y^=-52.17214255248879
y_pred = coef * (-0.67794537) + 100
print("예측값 : ",y_pred) # y : 39.34130801 : y^=39.34130756910188
print()

'''=====================================================================
방법 2 : LinerRegresstion 사용 - model을 생성함(실무에서 사용)
예측값을 줄때 학습한 차원값으로 값을 줘야함.
====================================================================='''
print('-'*20,'방법 2','-'*20)
from sklearn.linear_model import LinearRegression
# 값 가져오기
xx = x 
yy = y

# 모델 생성하기
model = LinearRegression()
fit_model = model.fit(xx, yy) # 최소제곱법으로 기울기, 절편을 반환함.
# 지원하는 함수 사용하기
print('기울기(slope) : ', fit_model.coef_)   # 89.47430739
print('절편(bias) : ', fit_model.intercept_) # 100.0
# 예측값 확인하는 함수
y_newpred = fit_model.predict(xx[[0]]) # 학습한 데이터가 2차원이면 2차원으로 값을 줘야함.
print("예측값 : ",y_newpred) # [-52.17214291]
y_newpred2 = fit_model.predict([[0.12345]]) 
print("예측값 : ",y_newpred2) # [111.04560325]
y_newpred3 = fit_model.predict([[0.12345], [0.3], [0.5]]) 
print("예측값 : ",y_newpred3) # [111.04560325 126.84229222 144.7371537 ]
print()

'''=====================================================================
방법 3 : ols 사용 - model을 생성함
    잔차제곱합(RSS)을 최소화하는 가중치 벡터를 행렬 미분으로 구하는 방법.
    ols는 1차원을 써야한다.

    LinearRegression과 똑같은데 summary의 유무차이임.
    선형회귀에서 보고서를 쓰기에 아주 훌륭한 정보를 제공함.↴(인과관계 판단 수단)
    P>|t| < α 이므로 인과관계가 있다고 판단할 수 있다. x1유의한 독립변수다.
    선형회귀 coef(계수)는 독립변수와 종속변수의 관계를 표시.
====================================================================='''
print('-'*20,'방법 3','-'*20)
import statsmodels.formula.api as smf

# ols가 원하는 형태 만들기
print(xx.ndim) # 2차원
x1 = xx.flatten() # 차원축소 - xx.flatten() or xx.ravel()
print(x1.ndim) # 1차원
y1 = y
data = np.array([x1, y1])
df = pd.DataFrame(data.T, columns=['x1','y1'])
print(df.head(3))

# 모델 생성하기
model2 = smf.ols(formula="y1 ~ x1", data=df).fit()
# summary() 사용하기 - 훌륭해~! report용 이런 결과를 주는건 ols밖에 없어.
print(model2.summary())
'''
                            OLS Regression Results
==============================================================================
Dep. Variable:                     y1   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 1.905e+32
Date:                Fri, 03 Apr 2026   Prob (F-statistic):               0.00
Time:                        11:01:33   Log-Likelihood:                 1460.6
No. Observations:                  50   AIC:                            -2917.
Df Residuals:                      48   BIC:                            -2913.
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
            coef(계수) std err(표준오차) t       P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept(d) 100.0000  7.33e-15   1.36e+16      0.000     100.000     100.000
x1       (w) 89.4743   6.48e-15   1.38e+16      0.000      89.474      89.474
==============================================================================
Omnibus:                        7.616   Durbin-Watson:                   1.798
Prob(Omnibus):                  0.022   Jarque-Bera (JB):                8.746
Skew:                           0.516   Prob(JB):                       0.0126
Kurtosis:                       4.770   Cond. No.                         1.26
==============================================================================
'''
print('기울기 :',model2.params['x1'])       # (w) 89.47430739278903
print('절편  :',model2.params['Intercept']) # (d) 99.99999999999999
print()

# 예측값 확인
new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]}) # 기존자료 검증
print("예측값 : ",model2.predict(new_df)) # -52.172143, 39.341308
new_df2 = pd.DataFrame({'x1':[0.1234, 0.2345]}) # 새로운 자료 검증
print("예측값2 : ",model2.predict(new_df2)) # 111.041130, 120.981725
print()