'''
Logistic Linear Regression(로지스틱회귀분석)
    선형결합을 logit(log odds)으로 해석하고, 이를 sigmoid함수를 통해 확률로 변환
    이항분류(다항도 가능-기능이 약함) 독립변수(연속형), 종속변수(범주형)인 지도학습
    로지스틱회귀 분석을 근거로 뉴럴넷의 뉴런에서 사용함.
    
    logit()은 변환 함수
    glm()은 logit을 포함한 전체 모델이다

선형회귀는 R²설명계수로 모델의 성능을 설명함.
로지스틱회귀는 정확도로 모델의 성능을 설명함
'''
# mtcars dataset 사용
import statsmodels.api as sm

mtcarsdata = sm.datasets.get_rdataset('mtcars')
# print(mtcarsdata.keys())# dict_keys(['data', '__doc__', 'package', 'title', 'from_cache', 'raw_data'])

mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars.info())

# 데이터 추출 - 연비(mpg)와 마력수(hp)에 따른 변속기(am - 수동, 자동) 분류
mtcar = mtcars.loc[:,['mpg','hp','am']]
print(mtcar.head(2))
print(mtcar.am.unique()) # [1(수동) 0(자동)]

#================================================================================================
# 모델 작성 방법 1 : logit()
#================================================================================================
import statsmodels.formula.api as smf
# from statsmodels.formula.api import logit
import numpy as np

formula = 'am ~ hp+mpg' # am 범주형 ~ hp+mpg 연속형

result = smf.logit(formula=formula, data=mtcar).fit()
print(result.summary())
print()

# 예측값 확인
# print('예측값 :', result.predict())
pred = result.predict(mtcar[:10])
print('예측값 :',pred.values) # [0.25004729 0.25004729 0.55803435 ....
print('예측값 :',np.around(pred.values)) # [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
# 범주형으로 출력 - 0.5 기준으로 0, 1이 출력됨.
print('실제값 :',mtcar['am'][:10].values) # [1 1 1 0 0 0 0 0 0 0] 
print()
'''
선형회귀는 R²설명계수
로지스틱회귀는 정확도로 모델의 성능을 설명함
'''

# 수치에 대한 집계표 확인
print("수치에 대한 집계표(Confusion matrix, 혼돈행렬, 혼동행렬) 확인--------------------")
# glm은 지원하지 않는다.
conf_tab = result.pred_table()
print(conf_tab)
'''
[[16.  3.]
[ 3. 10.]]
'''
# 현재 모델의 분류 정확도 확인1 - Confusion matrix이용
# print('분류 정확도 :', (16+10) / len(mtcar))
print('분류 정확도 :', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar)) # 0.8125 

# 모듈로 현재 모델의 분류 정확도 확인2 - accuracy_score이용
from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcar)
print('분류 정확도 :', accuracy_score(mtcar['am'], np.around(pred2))) # 0.8125
print()

#================================================================================================
# 모델 작성 방법2 : glm() - 일반화된 선형모델
#
#   Binomial() : 이항분포, Gaussian() : 정규분포-디폴트
#================================================================================================
# 모델 생성하기
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit() 
print(result2.summary())
glm_pred = result2.predict(mtcar[:10])
print("glm 예측값 :", np.around(glm_pred.values))
print('실제값 :',mtcar['am'][:10].values)

# 분류 정확도 확인
glm_pred2 = result2.predict(mtcar)
print('glm 분류 정확도 :', accuracy_score(mtcar['am'], np.around(glm_pred2))) # glm 분류 정확도 : 0.8125
print()

# logit()은 변환 함수, glm()은 logit을 포함한 전체 모델이다. 

# 새로운 값으로 분류 작업하기
print("새로운 값으로 분류 작업하기 --------------------------------------------")
import pandas as pd
newdf = pd.DataFrame()
newdf['mpg']=[10, 30, 120, 200]
newdf['hp'] = [100, 110, 80, 130]
print(newdf)
new_pred = result2.predict(newdf)
print('새로운 값 예측결과(np.around) :', np.around(new_pred.values)) #  [0. 1. 1. 1.]
print('새로운 값 예측결과(np.rint) :', np.rint(new_pred.values))
# [1(수동) 0(자동)]

