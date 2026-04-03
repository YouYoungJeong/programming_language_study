'''
회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 
심심하면 해보기 => statsmodels ols(), LinearRegression 사용
나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 
하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
- 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 
회귀분석 모델을 작성한 후에 예측하시오.
- 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 
회귀분석 모델을 작성한 후에 예측하시오.

참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 
이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  
'''
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import statsmodels.api as sm
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

data = pd.read_csv("lin_quiz1.csv")
print(data.info())
print(data.describe())

'''===================================================================
데이터 정제하기
======================================================================'''
# nan값 확인후 대체값 넣기
# 조건 - 결측치는 해당 칼럼의 평균 값을 사용
# print(data.isnull().sum()) # 지상파 na값 1 한개 존재
data['지상파'] = data['지상파'].fillna(data['지상파'].mean())
print(data.지상파.isnull().sum()) # 확인

# 이상치 확인 - 운동col 이상치 확인.
# sns.boxplot(data)
# plt.show()

# 이상치 처리하기
# 조건 - 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다
print(len(data))
data.drop(data[data['운동']>10].index, inplace=True)
print(len(data))

'''===================================================================
데이터 상관관계 확인 -0.8655346605559782(강한 음의 상관관계)
지상파 시청 시간(x)을 입력하면 어느 정도의 운동 시간(y)을 갖게 되는지
======================================================================'''
x = data.지상파
y = data.운동
print("데이터 상관관계 확인 :",np.corrcoef(x, y)[0,1])
print()

'''===================================================================
데이터 회귀분석 model1 = make_regression사용
======================================================================'''
print('-'*20,' model1 = make_regression사용','-'*20)
model1 = stats.linregress(x, y)
# print(model1)

print("기울기 :",model1.slope)
print("절편  :",model1.intercept)
print("p값(인과관계 확인)  :",model1.pvalue) # 6.347e-05 : 인과관계 有
# 회귀선 수식 y^ = -0.6684550167105406x + 4.709676019780582
print(f"회귀선 수식 y^ = {(model1.slope)}x + {(model1.intercept)})")
print()

# 예측하기
print('-----예측하기-----')
pred_value1 = [2.2, 3.3, 4.4]
for pred in pred_value1:
    print(f"지상파시청 예측시간 {(pred)}일때 운동시간: ", (model1.slope)*pred + model1.intercept)
    

# 시각화(추세선)
# 회귀선 수식 y^ = (model.slope*x + model.intercept)
# plt.scatter(x, y)
# plt.plot(x, (model.slope*x + model.intercept), c='r')
# plt.show()
print()

'''===================================================================
데이터 회귀분석 model2 : LinerRegresstion 사용
======================================================================'''
print('-'*20,' model2 : LinerRegresstion 사용','-'*20)

# 데이터 추출하기(차원변경)
# print(x.ndim) # 1차원
x1 = data['지상파'].values.reshape(-1, 1)
# print(x1.ndim) # 2차원
# print(x1)

# 예측할 값 생성
pred_value2 = np.array([2.2, 3.3, 4.4]).reshape(-1, 1)
# print(pred_value2)

# 모델생성
model2 = LinearRegression()
# 훈련
fit_model2 = model2.fit(x1, y) 
# w, d 확인하기
print('기울기(slope) : ', fit_model2.coef_)   
print('절편(bias) : ', fit_model2.intercept_)

# y^값 저장
y_pred2 = fit_model2.predict(pred_value2)

# 예측하기
print('예측하기(x = [2.2, 3.3, 4.4]) : ', y_pred2)
# [2.2, 3.3, 4.4] , [3.23907498 2.50377446 1.76847395]
print()

'''===================================================================
데이터 회귀분석 model3 : ols 사용
======================================================================'''
print('-'*20,' model3 : ols 사용','-'*20)

# 모델 생성하기
model3 = smf.ols(formula="운동 ~ 지상파", data=data).fit()
print('기울기 :',model3.params['지상파'])   # (w) 
print('절편  :',model3.params['Intercept']) # (d) 
print()

print('-----예측하기(x = [2.2, 3.3, 4.4])-----')
pred_value3 = pd.DataFrame({'지상파':[2.2, 3.3, 4.4]})
print("예측값 : \n",model3.predict(pred_value3).values) 

# 결론 티비보는 시간이 늘면 운동시간 감소