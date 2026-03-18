"""
배열 연산(벡터(b) y = ax + b)
배열 = Matrix
"""
import numpy as np
# 우선순위
x_1 = np.array([[1, 2],[3, 4]], dtype=np.float32) # float32
x_2 = np.array([[1, 2],[3, 4]]) # int64

# 2차원 배열
x = np.array([[1., 2],[3, 4]]) # float32
print(x)

# 1차원 배열
y_1 = np.arange(5,9)
print(y_1)

# 1차원 배열 => 2차원 배열로 바꿔주기 (reshape)
y = np.arange(5,9).reshape((2,2))

# 배열 타입 바꾸기 (y.astype)
y = y.astype(np.float32)
print(y)
print("-"* 50)

# 더하기
print(x + y)            # python의 연산자 또는 함수 (두개중 상대적으로 속도가 느림.)
print(np.add(x, y))     # numpy의 함수(유니버셜함수 : Ufnc),(상대적으로 속도가 빠름)

# 빼기
print(x - y)
print(np.subtract(x, y))

# 곱하기
print(x * y)
print(np.multiply(x, y))

# 나누기
print(x / y)
print(np.divide(x, y))
print("-"* 50)

""" 
내적 (행렬 곱)
dot은 numpy 모듈의 함수나 배열 객체의 인스턴트 메소드로 사용이 가능
"""
# 1차원 * 1차원
v = np.array([9,10])
w = np.array([11,12])
print(v * w) # 요소별 곱셈 9*11, 10*12
print(np.multiply(v, w))

# 두 벡터의 내적을 계산 - 행렬곱 - 안의 값을 벡터로 취급하고 벡터연산을 함.
# 내적 : 스칼라(크기만 있고 방향은 없음.)
print(v.dot(w))     # 9 * 11 + 10 * 12
print(np.dot(v, w)) 