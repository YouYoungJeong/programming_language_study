import numpy as np
"""1) step1 : array 관련 문제 https://cafe.daum.net/flowlife/SBU0/10
 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 
 각 행 단위로 합계, 최댓값을 구하시오."""

mun1 = np.random.randn(20).reshape(5,4)
# print(np.sum(abc, axis=0))
for i in range(4):
    print(f"{i+1}행 합계   : {np.sum(mun1[i], axis=0)}")
    print(f"{i+1}행 최댓값 : {np.max(mun1[i], axis=0)}")
"""--------------------------------------------"""
for i in mun1:
    print(f"행 합계   : {np.sum(i)}")
    print(f"행 최댓값 : {np.max(i)}")
    i += 1

print()
print("-"*15,"문제 2-1","-"*15)
"""step2 : indexing 관련문제,
문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오."""
mun2_1 = np.zeros((6,6))
print(mun2_1)

print()
# 조건1> 36개의 셀에 1~36까지 정수 채우기
mun2_1 = np.arange(1,37).reshape(6,6) 
print(mun2_1)
cnt = 0
for i in range(6):
    for j in range(6):
        cnt += 1
        mun2_1[i, j] = cnt
print(mun2_1)


# 출력할때 차원에서 떨어져나와.
print()
# 조건2> 2번째 행 전체 원소 출력하기 
print(np.array(mun2_1)[1, :]) # == [1]

print()
# 조건3> 5번째 열 전체 원소 출력하기
print(np.array(mun2_1)[:, 4])

print()
# 조건4> 15~29 까지 아래 처럼 출력하기
"""
[[15.16.17.]
[21.22.23]
[27.28.29.]]
"""
print(np.array(mun2_1)[2:5, 2:5])

"""--------------------------------------------"""
print()
print("-"*15,"문제 2-2","-"*15)
''' 문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.
조건1> 20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 
두 번째 열부터는 1씩 증가시켜 원소 저장하기'''
mun2_2 = np.zeros((6, 4))
mun2_rand = np.random.randint(20,100,6)
mun2_rand_li = list(mun2_rand)

for i in range(len(mun2_2)):
    # 시작값부터 1씩 증가하는 4개의 값 생성 arange(a,a+4)
    row_val = np.arange(mun2_rand[i], mun2_rand[i] + 4)
    #print(row_val)
    # 행에 저장
    mun2_2[i] = row_val
    #print(mun2_2)
print(mun2_2)
"""--------------------------------------------"""
print("-"*20)
for i in range(len(mun2_2)):
    num = mun2_rand_li.pop(0)
    for col in range(len(mun2_2[0])):
        mun2_2[i][col] = num
print(mun2_2)

print()
# 조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
mun2_2[0]=np.array([1000]*4)    # == mun2_2[0][:]
mun2_2[5]=np.array([6000]*4)    # == mun2_2[-1][:]
print(mun2_2)


print()
print("-"*15,"문제 3","-"*15)
"""
3) step3 : unifunc 관련문제
표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
배열 요소의 누적합을 출력하시오.
"""
# 전부 기술 통계량을 출력하는것(추론이 X)
mun3 = np.random.randn(20).reshape(4,5)
print(mun3)
print('평균 :', np.mean(mun3)) # numpy-> np.mean(mun3) == mun3.mean <-python module
print('합계 :', np.sum(mun3))
print('표준편차 :', np.std(mun3))
print('분산 :', np.var(mun3))
print('최댓값 :', np.max(mun3))
print('최솟값 :', np.min(mun3))
print('요소값 누적합 :', np.cumsum(mun3))
# 정규분포도 사분위수 출력
print('1사분위 수 :', np.percentile(mun3, 25))
print('2사분위 수 :', np.percentile(mun3, 50))
print('3사분위 수 :', np.percentile(mun3, 75))

print()
print("-"*15,"문제 Q1","-"*15)
"""
Q1) 브로드캐스팅과 조건 연산
다음 두 배열이 있을 때,
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.
그 결과에서 값이 30 이상인 요소만 골라 출력하시오.
"""
a = np.array([[1], [2], [3]])   # 3 * 1
b = np.array([10, 20, 30])      # 1 * 3
conc = a * b # <- 알아서 브로드캐스팅이 일어남.
print(conc)
condi = np.array(conc <= 30)
print(conc[condi])

print()
print("-"*15,"문제 Q2","-"*15)
"""
Q2) 다차원 배열 슬라이싱 및 재배열
 - 3×4 크기의 배열을 만들고 (reshape 사용),  
 - 2번째 행 전체 출력
 - 1번째 열 전체 출력
 - 배열을 (4, 3) 형태로 reshape
 - reshape한 배열을 flatten() 함수를 사용하여 1차원 배열로 만들기
"""
mun5 = np.arange(1, 13).reshape(3,4)
print(mun5)
print(mun5[1])
print(mun5[:,0])

resh = mun5.reshape(4,3)
print(resh)

print(resh.flatten())

print("-"*15,"문제 Q3","-"*15)
"""Q3) 1부터 100까지의 수로 구성된 배열에서 
3의 배수이면서 5의 배수가 아닌 값만 추출하시오.
그런 값들을 모두 제곱한 배열을 만들고 출력하시오.
"""
mun6 = np.arange(1,101)
condi2 = (mun6%3 == 0) & (mun6 %5 != 0)
# print(np.where(condi2, mun6,np.nan))
print(mun6[condi2]**2)

print()
print("-"*15,"문제 Q4","-"*15)
"""
Q4) 다음과 같은 배열이 있다고 할 때,
arr = np.array([15, 22, 8, 19, 31, 4])
값이 10 이상이면 'High', 그렇지 않으면 'Low'라는 문자열 배열로 변환하시오.
값이 20 이상인 요소만 -1로 바꾼 새로운 배열을 만들어 출력하시오. (원본은 유지)
힌트: np.where(), np.copy()
"""
mun7 = np.array([15, 22, 8, 19, 31, 4])
print(np.where(mun7 >= 10, 'High','Low' ))
mun7_c= np.copy(mun7)
mun7_c[mun7_c >= 20] = -1
print(mun7_c)

"""
Q5) 정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개를 만들고, 상위 5% 값만 출력하세요.
힌트 :  np.random.normal(), np.percentile()
"""

data = np.random.normal(loc=50, scale=10, size=1000)
print(data)
threshold = np.percentile(data, 95)
top5 = data[data> threshold]
print(top5)