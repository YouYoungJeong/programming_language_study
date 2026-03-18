import numpy as np
"""1) step1 : array 관련 문제 https://cafe.daum.net/flowlife/SBU0/10
 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 
 각 행 단위로 합계, 최댓값을 구하시오."""

mun1 = np.random.randn(20).reshape(5,4)
# print(np.sum(abc, axis=0))
for i in range(4):
    print(f"{i+1}행 합계   : {np.sum(mun1[i], axis=0)}")
    print(f"{i+1}행 최댓값 : {np.max(mun1[i], axis=0)}")


"""step2 : indexing 관련문제,
문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오."""
mun2_1 = np.zeros((6,6))
print(mun2_1)
# 조건1> 36개의 셀에 1~36까지 정수 채우기
mun2_1 = np.arange(1,37).reshape(6,6) 
print(mun2_1)
# 조건2> 2번째 행 전체 원소 출력하기 
print(np.array(mun2_1)[1])
# 조건3> 5번째 열 전체 원소 출력하기
print(np.transpose(mun2_1)[4])
# 조건4> 15~29 까지 아래 처럼 출력하기 -> ?
"""
[[15.16.17.]
[21.22.23]
[27.28.29.]]
"""
print(mun2_1.reshape(-1)[5:10:2])
''' 문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.
조건1> 20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 
두 번째 열부터는 1씩 증가시켜 원소 저장하기'''
mun2_2 = np.zeros(6,4)
