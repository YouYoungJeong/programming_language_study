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
# 내적(행렬곱) : 스칼라(크기만 있고 방향은 없음.)
print(v.dot(w))         # 9 * 11 + 10 * 12
print(np.dot(v, w))     # 1차원이기 때문에 스칼라로 출력
print(np.dot(x, v))     # 2차원이기 때문에 값이 2개(cos값)  
print('-'*50)

# 배열 계산 함수
print(x)
print(np.mean(x), " ", np.var(x), "\n")
print(np.max(x), " ", np.min(x), "\n")          # 값을 반환
print(np.argmax(x), " ", np.argmin(x), "\n ")   # argmax인덱스 반환
print(np.cumsum(x)) # 누적합
print(np.cumprod(x)) # 누적곱
print('-'*50)

names1 = np.array(['tom','james','tom','oscar'])
names2 = np.array(['tom','page','john'])
print(np.unique(names1)) # 중복 제거
print(np.intersect1d(names1, names2)) # 교집합
print(np.intersect1d(names1, names2, assume_unique=True)) # 교집합 옵션 설정 assume_unique : 중복허용
print(np.union1d(names1, names2)) # 합집합
print('-'*50)

'''전치 : 2차원 배열에서 행과 열의 위치를 바꿈'''
print(x)
print(x.T)
print(x.transpose())
print(x.swapaxes(0,1)) #swapaxes(행,열)
print('-'*50)

'''
Broadcasting : 크기가 다른 배열간의 연산
작은 배열을 여러번 반복해 연산
'''
x = np.arange(1, 10).reshape(3, 3)
y = np.array([1, 0, 1])
print(x)
print(y)
print(x + y)

# x데이터를 txt파일로 만들어줌.
np.savetxt("my.txt", x) # 배열 file i/o
# np.loadtxt("my.txt") # 불러오기