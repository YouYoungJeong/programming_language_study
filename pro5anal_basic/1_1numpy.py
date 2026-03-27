"""
numpy의 ndraary는 단순한 배열이라기 보다,
백터 / 행렬 연산도 가능한 다차원 수치 데이터 구조다
"""
import numpy as np

# python 의 list 여러가지 종류의 값이 들어갈 수 있다
ss = ["tom", "james", "oscar", 1, True]
print(ss, " ", type(ss)) # ['tom', 'james', 'oscar'] list 

# ndarray는 다 동일한 타입으로 변환시킴, 
# 같은 타입의 자료로만 구성됨.
ss2 = np.array(ss)
print(ss2, " ", type(ss2)) # ['tom' 'james' 'oscar']  numpy.ndarray ','없이 나옴

# Why Python is Slow :  https://cafe.daum.net/flowlife/RUrO/118
# list의 전체 값은, 별도의 객체로 다 기억함
# 각각의 요소들이 서로다른 주소를 가지고 있음.
li = list(range(1, 10))
print(li)
print(li[0], " ", id(li[0]))
# 각 값 즉 요소에 10을 곱하고싶은데
# print(li * 10) # 리스트 묶음을 10번 반복하라는 뜻
# for문을 돌려야만 값이 나옴
for i in li:
    print(i * 10, end=' ')
print("\n","-"*50)


# numpy 사용
# 각 요소의 주소가 같음.
num_arr = np.array(li)
print(num_arr[0], num_arr[1],' ' ,id(num_arr[0]) ,id(num_arr[1]) )
# 벡터화 연산이 가능.
print(num_arr * 10)
print("\n","-"*50)


"""
ndarray
동일 타입만 취급
여러 타입의 자료가 입력되면 상위 타입으로 자동변환
int < flaot < complex(복소수) < string
"""
a = np.array([1,2,3])
print(a, type(a)) # 전체 정수
b = np.array([1,2,3.5], dtype="float32") # dtype="float32"명시적으로 쓸 수 있다. 안써도 자기가 알아서 바뀜
print(b, type(b)) # 전체 실수
print("\n","-"*50)

# 2차원 배열
c = np.array([[1,2,3],[4,5,6]])
print(f"c\n{c}, \nc.shape = {c.shape} \nc[0,0] = {c[0,0]} c[[0]] = {c[[0]]}")

# 단위 행렬
# 2행 2열 만들기
d = np.zeros((2,2))
print(d)

# 1로 만들어진 2행 2열 만들기
e = np.ones((2,2))
print(e)

# 주 대각이 1일 단위행렬
f = np.eye(10)
print(f)

# 난수 생성
print(np.random.rand(5))        # rand(균등 분포) 
print(np.random.randn(5))       # randn(정규 분포) mean=0, std=1
np.random.seed(0) # 난수생성 고정
print(np.random.randn(2,3))

# 리스트 생성
print(list(range(0,10)))
print(np.arange(10))
print("\n","-"*50)

# 인덱싱/슬라이싱
a = np.array([1,2,3,4,5])
print(a , " " , a[1]) # 인덱싱
print(a[1:4])       # 슬라이싱 1:4 = 1≤a[index]＜4
print(a[1:])        # 1번부터 끝까지
print(a[1:5:2])     # 1이상 5미만인 값중 2step
print(a[-2:])       # 뒤에서부터 -1, -2
b = a
print(a[0],b[0])
# 주소 치환
b[0] = 88
print(a[0],b[0])

#copy 카피. 복사본 생성
c = np.copy(a)
c[0] = 33
print(a[0], b[0], c[0])