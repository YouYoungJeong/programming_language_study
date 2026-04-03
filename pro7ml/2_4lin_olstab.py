'''
단순선형회귀 - ols사용하고 Regression Result의 이해(.summary())
참고 : https://cafe.daum.net/flowlife/SBYs/3
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import koreanize_matplotlib
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv")
print(df.head(3))
print(df.corr())
# 만족도(종속)와 적절성(독립)이 상관관계가 가장 높음

# 모델 생성
model = smf.ols(formula='만족도 ~ 적절성',data=df).fit()
print(model.summary())
'''
                            OLS Regression Results
==============================================================================
Dep. Variable:                 만족도   R-squared(**설명력) - 독립변수가 하나일때
Model:                            OLS   Adj. R-squared(수정된 결정계수) -독립변수가 하나 이상일때
Method:                 Least Squares   F-statistic - F값 = t값**2
Date:                Fri, 03 Apr 2026   Prob (F-statistic) - 이 모델에 대한 p-value(모델의 유의함을 표시(p<α 쓸모있는 모델))
Time:                        14:42:15   Log-Likelihood:                -207.44
No. Observations:                 264   AIC:                             418.9
Df Residuals:                     262   BIC:                             426.0
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                coef    std err    t(codf/str err)  P>|t|(각각의 독립변수에대한 유의값p<α)      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.7789      0.124      6.273      0.000       0.534       1.023
적절성         0.7393      0.038     19.340      0.000       0.664       0.815
==============================================================================
Omnibus:                       11.674   Durbin-Watson:                   2.185
Prob(Omnibus):                  0.003   Jarque-Bera (JB):               16.003
Skew:                          -0.328   Prob(JB):                     0.000335
Kurtosis:                       4.012   Cond. No.                         13.4
==============================================================================
t : 기울기(β = coef)/표준오차(std err)
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''
# np.set_printoptions(suppress=True, precision=10)
# pd.options.display.float_format = '{:.10f}'.format
print('parameters(회귀계수) :', model.params)
print('R-squared :', model.rsquared) # 0.5880630629464404
print('p-values :', model.pvalues) # 2.235345e-52

print('예측값 : ', model.predict()[:5])
print('실제값 : ', df.만족도[:5].values)
# 예측값 :  [3.73596305 2.99668687 3.73596305 2.25741069 2.25741069]
# 실제값 :  [3 2 4 2 2]

plt.scatter(df.적절성, df.만족도)
# plt.scatter(df.적절성, model.predict())
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)
plt.plot(df.적절성,slope*(df.적절성)+intercept, c='b')
plt.show()


