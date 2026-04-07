'''
sigmoid function 적용 연습
    로지스틱회귀에서는 wx + b자체는 logit한 값이다.
    log( p / 1-p ) = wx + b 라고 정의가 되어있다.
    그러므로 z = wx + b => sigmoid(z) => p(0 ~ 1)
'''

# sigmoid 함수 수식으로 반환된 값 확인
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
np.set_printoptions(suppress=True, precision=10)
import math
def sigmoidFunc(num):
    # num  = 로짓변환된 함수
    return 1 / (1 + math.exp(-num))
print(sigmoidFunc(3))   # 0.9525741268224334 = 1
print(sigmoidFunc(1))   # 0.7310585786300049 = 1
print(sigmoidFunc(-5))  # 0.0066928509242848 = 0
print(sigmoidFunc(-10)) # 4.539786870243e-05 = 0
print()

print("logit변환된(가정) 값으로 시그모이드 함수 통과 후 그 결과를 시각화하기")
x = np.linspace(-10, 10, 50) # 입력자료(연속형)
# print(x)

# 선형결합(이미 logit값)
w = 1.5
b = -2
z = w*x + b
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
p = sigmoid(z) # 확률값을 얻게됨.
print('p: \n',p)

# 일부 값 보기
print("x[:3]", np.round(x[:3],3))         # [-10.           -9.592           -9.184]
print("z[:3] (logit)", np.round(z[:3],3)) # [-17.           -16.388         -15.776]
print("p[:3] (확률)", p[:3])              # [0.0000000414 0.0000000764 0.0000001409]
# 전부 음수가 나왔기 때문에 0.5보다 작다

# 시각화하기
plt.figure(figsize=(8,5))
plt.plot(x, p, label='sigmoid(z)', color='blue')
plt.axhline(0.5, color='r', linestyle='--', label='axhline(0.5)')
plt.title('z = wx + b -> sigmoid -> 확률값')
plt.xlabel('x(입력값)')
plt.ylabel('p(확률값)')
plt.grid(True)
plt.legend()
plt.show()


