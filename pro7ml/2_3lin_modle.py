'''=====================================================================
방법 4 : make_regression 사용 - model을 생성하지 않음
    predict method 를 지원하지 않음
====================================================================='''
print('-'*20,'방법 4','-'*20)
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 예측
score_iq = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv")
print(score_iq.head(3))
print(score_iq.info())

# 데이터 정제
x = score_iq.iq
y = score_iq.score
print(x[:3])
print(y[:3])

# 상관관계 확인 - 매우 강한 상관관계
print("상관계수 :", np.corrcoef(x, y)[0, 1]) # 0.8822203446134701
print(score_iq[['iq','score']].corr())

# 상관관계 시각화
# plt.scatter(x, y)
# plt.show()

# 단순 선형회귀분석 (인과관계가 있다는 가정하고 진행)
model = stats.linregress(x, y)
print(model)
print()
'''
LinregressResult(
        slope=np.float64(0.6514309527270075), - 기울기
        intercept=np.float64(-2.8564471221974657), - 절편
        rvalue=np.float64(0.8822203446134699), 
        pvalue=np.float64(2.8476895206683644e-50),  - 인과관계가 있는게 맞구나
        stderr=np.float64(0.028577934409305443),
        intercept_stderr=np.float64(3.546211918048538) 
    )
'''
print("기울기 :",model.slope)
print("절편  :",model.intercept)
print("p값(인과관계 확인)  :",model.pvalue)

# 시각화(추세선)
# 회귀선 수식 y^ = (model.slope*x + model.intercept)
# plt.scatter(x, y)
# plt.plot(x, (model.slope*x + model.intercept), c='r')
# plt.show()

# 점수 예측
# print("전체 값에 대한 점수 예측 :", np.polyval([model.slope, model.intercept], np.array(score_iq.iq)))

new_df = pd.DataFrame({'iq':[55,66,77,88,150]})
print("new_df 값에 대한 점수 예측 :\n", np.polyval([model.slope, model.intercept], new_df))