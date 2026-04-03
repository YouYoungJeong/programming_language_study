'''
최소 자승법 (최소제곱법, Least Square Method)
최소제곱해 선형 행렬 방정식으로 얻기
'''
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.5, 2.1])

# plt.scatter(x, y)
# plt.grid(True)
# plt.show()

# 이차원 데이터 생성
A = np.vstack([x, np.ones(len(x))]).T 
print(A)

# 본래 데이터를 직선으로 표현하기 위해 선형대수학 이용
import numpy.linalg as lin # 최소제곱법 함수 호출
# y = wx + b 의 w와 b구하기
weight, bias = lin.lstsq(A, y)[0] # 최소제곱법 연산(내부적으로 편미분 사용)
print(weight, bias) # 0.96(w:기울기) -0.9899999999999998(b:절편)
print()
# 회귀식이 완성 => y^ = 0.96*x + (-0.9899999999999998)
# y^ = (weight)*x + bias

print(0.96*0 + (-0.9899999999999998)) # 실제값 -1,  예측값 -0.989
print(0.96*1 + (-0.9899999999999998)) # 실제값 0.2, 예측값 -0.0299
print(0.96*2 + (-0.9899999999999998)) # 실제값 0.5, 예측값 0.93
print(0.96*3 + (-0.9899999999999998)) # 실제값 2.1, 예측값 1.89


plt.scatter(x, y, marker="o", label="실제값")
plt.plot(x, (weight*x + bias),'r', label='최적화된 선형직선(추세선)')
plt.grid(True)
plt.legend()
plt.show()

# 경험하지 않은 x값에 대한 y값은? 을 구할 수가 있다.(추세선이 완성이 되었기 때문)
x = 1.2345
yhat = weight*x + bias
print("예측결과 : ", yhat) # 예측결과 :  0.19512000000000018

x = 7.654321
yhat = weight*x + bias
print("예측결과 : ", yhat) # 예측결과 :  6.358148160000001
