'''
회귀분석 모형의 적절성을 위한 선행조건 5가지
    다중회귀분석 모형의 적절성을 위한 선행조건
        1)+2)+3)
        4)독립성 : 잔차가 자기상관(인접 관측치의 오차가 상관됨)이 있는지 확인
            Durbin-Watson : 잔차의 자기상관(autocorrelation) 검정 지표. 
                잔차들이 서로 독립적인가? 시간 흐름 데이터에서 중요 (시계열)
                값의 범위는 0 ~ 4 이고   
                2이면 정상 (자기상관 없음).   
                < 2이면 양의 자기상관,  
                > 2이면 음의 자기상관
                model.summary()로 확인 가능하다.
        5)다중공선성 : 다중회귀 분석 시 독립변수간에 강한 상관관계가 있으면 안된다.
            VIF(variance_inflation_factor, 분산 팽창 지수, 분산 인플레 요인) 
            : 값이 10을 넘으면 다중공선성이 발생하는 변수라고 할 수 있다.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import shapiro # 정규성 확인
import statsmodels.api as sm


# 데이터 가져오기
df =  pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv")
# print(df.head(3)) # 확인
print(df.info())
# object 칼럼 제거 연속형데이터만 사용할것.
df = df.drop([df.columns[6], df.columns[9], df.columns[10]],axis=1)
print(df.corr()) # 상관관계 낮은 칼럼 제거하기- (CompPrice, Population, Education)
print()

# 모델생성 - x:평균소등, 광고, 가격, 나이,
lm = smf.ols(formula='Sales ~ Income+Advertising+Price+Age', data=df).fit()
print(lm.summary())
print()
'''
Prob (F-statistic):  1.33e-38     - 의미 있는 모델
Adj. R-squared: 0.364             - 15%이상이면 쓰니까 쓸만한거야
독립변수 전체 P>|t| < 0.05        - 이므로 유의한 설명변수임.
'''
# 선형회의 모델의 적절성 조건 체크후 모델 사용
print("="*20,'모델의 적절성 확인하기','='*20)
df_lm = df.iloc[:, [0,2,3,5,6]]
print(df_lm.head(1))
print()

# 잔차항 구하기
print('잔차항 확인하기')
fitted = lm.predict(df_lm)
residual = df_lm['Sales'] - fitted
print('residual : ', residual[:3])
print('residual mean : ', np.mean(residual)) # -1.412e-15
print()
# ====================================================================================
# 1) 잔차의 정규성 검정
# ====================================================================================
print('-'*10,'잔차가 정규성을 따르는지 확인하기','-'*10)
# shapiro test, Q-Q plot사용
stat, p = shapiro(residual)
print(f'shapiro 통계량 : {stat:.4f}, p-value : {p:.4f}') 
# p-value : 0.2127 > α 이므로 정규성 만족
print("p-value 정규성 만족" if p > 0.05 else "p-value 정규성 위배")
print()

# Q-Q plot 시각화 - 정규성을 만족하고 있다.
# sm.qqplot(residual, line='s')
# plt.title("Q-Q Plot로 정규성 만족 확인")
# plt.show()

# ====================================================================================
# 2) 선형성 검정
# ====================================================================================
print('-'*10,'잔차가 선형성을 따르는지 확인하기 : 독립변수의 변화에 종속변수도 변화하나 특정한 패턴이 있으면 안됨','-'*10)
from statsmodels.stats.diagnostic import linear_reset # 선형성 확인 모델

reset_result = linear_reset(lm, power=2, use_f=True) # 설명값을 제곱, f값을 이용할거야
print(f"reset_result 결과(p) : {reset_result.pvalue:.5f}")
print("선형성 만족" if reset_result.pvalue > 0.05 else "선형성 위배")
print()

# 선형성 시각화 - 이정도의 선은 만족한다고 볼 수 있다.
# sns.regplot(x=fitted, y=residual, lowess=True, 
#             line_kws={'color':'magenta'},   # 선형성 시각화
#             scatter_kws={'color':'gray'},   # 잔차값 시각화 
#             )
# # 기준선 그리기
# plt.plot([fitted.min(), fitted.max()], [0,0], '--', color='blue')
# plt.xlabel('예측값(Fitted Values)')
# plt.ylabel('잔차(Residuals)')
# plt.title('선형성 확인용 잔차 그래프')
# plt.show()

# ====================================================================================
# 3) 등분산성(Homoscedasticity) 검정
# ====================================================================================
# 시각화는 선형성 시각화 참조
print('-'*10,'등분산성(Homoscedasticity) 검정 : 모든 x값에서 오차의 퍼짐정도(분산)가 일정해야한다.','-'*10)
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(resid=residual, exog_het=sm.add_constant(df_lm['Sales']))
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f'het_breuschpagan test 통계량: {bp_stat}, p-value:{bp_pvalue}')
print("등분산성 만족" if bp_pvalue > 0.05 else "등분산성 위배")
print()

# ====================================================================================
# 4) 독립성 검정
# ====================================================================================
print('-'*10,'독립성 검정 : 잔차가 자기상관(인접 관측치의 오차가 상관됨)이 있는지 확인','-'*10)
import statsmodels.api as sm

print('Durbin-Watson : ', sm.stats.stattools.durbin_watson(residual))
# 1.9314981270829592 이므로  2이면 정상 (잔차의 자기상관 없음)
print()

# ====================================================================================
# 5) 다중공선성 검정
# ====================================================================================
print('-'*10,'다중공선성 검정 : 다중회귀 분석 시 독립변수간에 강한 상관관계가 있으면 안된다','-'*10)
# VIF(variance_inflation_factor, 분산 팽창 지수, 분산 인플레 요인) 
# : 값이 10을 넘으면 다중공선성이 발생하는 변수라고 할 수 있다.
from statsmodels.stats.outliers_influence import variance_inflation_factor

df_ind = df[['Income', 'Advertising', 'Price', 'Age']] # 독립변수 목록
vifdf = pd.DataFrame()
vifdf['변수'] = df_ind.columns
vifdf['vif_values'] = \
    [variance_inflation_factor(df_ind.values, i) for i in range(df_ind.shape[1])]
print(vifdf) # 10을 초과하지 않았으므로 모두 만족
print()
'''      변수  vif_values
0       Income    5.971040
1  Advertising    1.993726
2        Price    9.979281
3          Age    8.267760'''

# 시각화하기
# sns.barplot(x='변수', y='vif_values', data=vifdf)
# plt.title('VIF')
# plt.show()

# ====================================================================================
# 모델 저장하기 
# ====================================================================================
# 유의한(의미있는) 모델이므로 생성된 모델을 파일로 저장하고 이를 재사용한다고 한다면
'''
방법 1 - pickle사용 (pickle은 binary로 i/o 해야하므로 번거롭다.)
import pickle
with open('carseat_model.pickle',mode='wb') as obj: # pickle로 저장
    pickle.dump(lm, obj) 
with open('carseat_model.pickle',mode='rb') as obj: # pickle로 읽기
    mymodel = pickle.load(lm, obj) 
    '''
# 방법2 - joblib (pickle보다 조금더 깔끔함)
import joblib
joblib.dump(lm,'carseat.model')         # 저장
mymodel = joblib.load('carseat.model')  # 읽기
# 이후부터는 'carseat.model'를 읽어 사용하면 됨. - lm은 없어도 됨.

print("새로운 값으로 Sales 예측")
# ['Income', 'Advertising', 'Price', 'Age']
new_df = pd.DataFrame({'Income':[35, 62], 
                    'Advertising':[6, 3],
                    'Price':[105, 88],
                    'Age':[32,55]})
pred = mymodel.predict(new_df)
print('Sales예측 결과 : ',pred.values) # [8.71289759 8.49715914]
