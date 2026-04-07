import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error 
from db import fetchall_ibsail

# 근무년수에 대한 연봉 pay(y), years(x)
def linearFunc(year):
    # db.py에서 직원테이블 받아 DataFrame생성
    data = fetchall_ibsail()
    df = pd.DataFrame(data)

    # 독립변수, 종속변수 추출
    x = df[['years']]   # 독립변수(2차원)
    y = df['pay']       # 종속변수
    
    # model생성
    model = LinearRegression().fit(x, y)
    
    # w. b 확인 소수점조절- 출력용
    slope= round(float(model.coef_[0]),4)
    intercept = round( model.intercept_,4)
    # print('slope(w) :',slope)
    # print("intercept(b) :", intercept)
    
    # 회귀식 생성
    sik = f"pay = {slope} * years + ({intercept})"
    
    # 예측 y^ 생성
    y_pred = model.predict(x) # 예측값

    # R² 결정계수
    r_score = round(r2_score(y, y_pred),2) * 100
    r_scoremsg = f"{r_score}%"

    # main에서 받은 year값 예측하기
    year = float(year)  # str -> float
    years =[[year]]     # 2차원
    pay_pred = int(model.predict(years)[0]) # 연봉 예측값 출력 소수점자르기
    
    # 예상연봉액이 음수인 경우 0 을 반환
    if pay_pred < 0:
        pay_pred = 0

    # R² , 회귀식,  연봉 예측값 반환
    return(r_scoremsg, sik, pay_pred) 

