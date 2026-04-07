'''
비선형회귀분석(Non-linear regression)
    70년대 미국 보스턴 시의 주택가격을 설명한 dataset
    보스톤 주택 가격 데이터는 회귀를 다루는 많은 기계 학습 논문에서 사용되었다
    참고 : 70년대 미국 보스턴 시의 주택가격을 설명한 dataset

회귀분석의 한 예로 scikit-learn 패키지에서 제공하는 주택가격을 예측하는 Dataset을 사용할 수 있다. 
이는 범죄율, 공기 오염도 등의 주거 환경 정보 등을 사용하여 
70년대 미국 보스턴 시의 주택가격을 표시하고 있다.
* 데이터 세트 특성 :
    : 인스턴스 수 : 506
    : 속성의 수 : 13 개의 숫자 / 범주 적 예측
    : 중간 값 (속성 14)은 대개 대상입니다
    : 속성 정보 (순서대로) :

CRIM   자치시(town) 별 1인당 범죄율
ZN 25,000   평방피트를 초과하는 거주지역의 비율
INDUS   비소매상업지역이 점유하고 있는 토지의 비율
CHAS   찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0)
NOX   10ppm 당 농축 일산화질소
RM   주택 1가구당 평균 방의 개수
AGE   1940년 이전에 건축된 소유주택의 비율
DIS   5개의 보스턴 직업센터까지의 접근성 지수
RAD   방사형 도로까지의 접근성 지수
TAX   10,000 달러 당 재산세율
PTRATIO   자치시(town)별 학생/교사 비율
B   1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함.
LSTAT   모집단의 하위계층의 비율(%)
MEDV   본인 소유의 주택가격(중앙값) (단위: $1,000)

['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

MEDV  (Y : 종속변수)
LSTAT (X : 독립변수)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.max_columns', None)

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data",
                header=None, sep=r'\s+', # 정규표현식 사용 공백으로 구분
                names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
# print(df.head(2))
# print(df.info())
# print(df.corr()) # MEDV(집값중앙값), LSTAT(하위계층비율) : -0.737663 강한음의상관관계

# 변수 추출
x = df[['LSTAT']].values   # 독립변수 2차원 데이터
y = df['MEDV'].values      # 종속변수 1차원 데이터
print(x[:3])
print(y[:3])



# 선형 모델 생성하기(단항을 통한 선형모델)
model = LinearRegression()
# 다항특성을 통한 모델 생성하기
# degree = 2
quad = PolynomialFeatures(degree=2)
x_quad = quad.fit_transform(x)
# print(x_quad[:3])
'''
x_quad(degree=2)
[[ 1.      4.98   24.8004]
    [ 1.      9.14   83.5396]
    [ 1.      4.03   16.2409]]
'''
# degree = 3
cubic = PolynomialFeatures(degree=3)
x_cubic = cubic.fit_transform(x)
# print(x_cubic[:3])
'''
x_cubic(degree=3)
[[  1.         4.98      24.8004   123.505992]
    [  1.         9.14      83.5396   763.551944]
    [  1.         4.03      16.2409    65.450827]]
'''

# 단순회귀
model.fit(x, y)
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis] # 그래프 표시용
y_lin_fit = model.predict(x_fit)
# print('y_lin_fit :',y_lin_fit)
model_r2 = r2_score(y, model.predict(x))
print('model_r2 :',model_r2)

# 2차
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
quad_r2 = r2_score(y, model.predict(x_quad))
print('quad_r2 :',quad_r2) # 0.640716

# 3차
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
cubic_r2 = r2_score(y, model.predict(x_cubic))
print('cubic_r2 :',cubic_r2) # 0.65784


# 초기 데이터 시각화 -> 음의 상관관계의 커브를 그리고 있다.
plt.scatter(x, y, label='초기 데이터')
plt.plot(x_fit, y_lin_fit, linestyle=":", label='linear fit(d=1), $R^2=%.2f$'%model_r2, c='aqua', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle="-", label='quad fit(d=2), $R^2=%.2f$'%quad_r2, c='g', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle="--", label='cubic fit(d=3), $R^2=%.2f$'%cubic_r2, c='magenta', lw=3)
plt.xlabel('하위 계층 비율')
plt.ylabel("주택가격 중앙값")
plt.legend()
plt.show()


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

# 1. 초기 데이터만
ax[0].scatter(x, y, label='초기 데이터')
ax[0].set_title("초기 데이터")
ax[0].set_xlabel("하위 계층 비율")
ax[0].set_ylabel("주택가격 중앙값")
ax[0].legend()

# 2. 초기 데이터 + 선형 추세선
ax[1].scatter(x, y, label='초기 데이터')
ax[1].plot(x_fit, y_lin_fit, linestyle=":", 
            label='linear fit(d=1), $R^2=%.2f$' % model_r2,
            c='aqua', lw=3)
ax[1].set_title("선형 추세선")
ax[1].set_xlabel("하위 계층 비율")
ax[1].set_ylabel("주택가격 중앙값")
ax[1].legend()

# 3. 초기 데이터 + 2차 다항 추세선
ax[2].scatter(x, y, label='초기 데이터')
ax[2].plot(x_fit, y_quad_fit, linestyle="-", 
            label='quad fit(d=2), $R^2=%.2f$' % quad_r2,
            c='g', lw=3)
ax[2].set_title("2차 다항 추세선")
ax[2].set_xlabel("하위 계층 비율")
ax[2].set_ylabel("주택가격 중앙값")
ax[2].legend()

# 4. 초기 데이터 + 3차 다항 추세선
ax[3].scatter(x, y, label='초기 데이터')
ax[3].plot(x_fit, y_cubic_fit, linestyle="--", 
            label='cubic fit(d=3), $R^2=%.2f$' % cubic_r2,
            c='magenta', lw=3)
ax[3].set_title("3차 다항 추세선")
ax[3].set_xlabel("하위 계층 비율")
ax[3].set_ylabel("주택가격 중앙값")
ax[3].legend()

plt.tight_layout()
plt.show()