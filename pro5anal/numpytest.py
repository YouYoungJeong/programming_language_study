import numpy as np
"""1) step1 : array 관련 문제 https://cafe.daum.net/flowlife/SBU0/10
 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 
 각 행 단위로 합계, 최댓값을 구하시오."""

mun1 = np.random.randn(20).reshape(5,4)
# print(np.sum(abc, axis=0))
for i in range(4):
    print(f"{i+1}행 합계   : {np.sum(mun1[i], axis=0)}")
    print(f"{i+1}행 최댓값 : {np.max(mun1[i], axis=0)}")

print()
print("-"*15,"문제 2-1","-"*15)
"""step2 : indexing 관련문제,
문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오."""
mun2_1 = np.zeros((6,6))
print(mun2_1)
# 조건1> 36개의 셀에 1~36까지 정수 채우기
mun2_1 = np.arange(1,37).reshape(6,6) 
print()
print(mun2_1)
# 조건2> 2번째 행 전체 원소 출력하기 
print(np.array(mun2_1)[1])
print()
# 조건3> 5번째 열 전체 원소 출력하기
print(np.transpose(mun2_1)[4])
print()
# 조건4> 15~29 까지 아래 처럼 출력하기
"""
[[15.16.17.]
[21.22.23]
[27.28.29.]]
"""
print(np.array(mun2_1)[2:5, 2:5 ])
print()
print("-"*15,"문제 2-2","-"*15)
''' 문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.
조건1> 20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 
두 번째 열부터는 1씩 증가시켜 원소 저장하기'''
mun2_2 = np.zeros((6, 4))
mun2_rad = np.random.randint(20,101,6)


# !!
for i in range(6):
    # 시작값부터 1씩 증가하는 4개의 값 생성 arange(a,a+4)
    row_values = np.arange(mun2_rad[i], mun2_rad[i] + 4)
    #print(row_values)
    # 행에 저장
    mun2_2[i] = row_values
    #print(mun2_2)
print(mun2_2)

# 조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
mun2_2[0]=np.array([1000]*4)
mun2_2[5]=np.array([6000]*4)
print(mun2_2)


print("-"*15,"문제 3","-"*15)
"""
3) step3 : unifunc 관련문제
표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
배열 요소의 누적합을 출력하시오.
"""
mun3 = np.random.rand(20).reshape(4,5)
print(mun3)
print('평균 :', np.mean(mun3))
print('합계 :', np.sum(mun3))
print('표준편차 :',np.std(mun3))
print('분산 :', np.var(mun3))
print('최댓값 :', np.max(mun3))
print('최솟값 :', np.min(mun3))
print('1사분위 수 :')
print('2사분위 수 :')
print('3사분위 수 :')
print('요소값 수 :', np.cumsum(mun3))


"""
Q1) 브로드캐스팅과 조건 연산
다음 두 배열이 있을 때,
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.
그 결과에서 값이 30 이상인 요소만 골라 출력하시오.
"""
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
conc = np.concatenate(a)*b
print(conc)
condi = np.array(np.concatenate(a)*b <= 30)
print(np.where(condi, conc, False))