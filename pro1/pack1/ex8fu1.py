# function : 여러개의 수행문을 하나의 이름으로 묶은 실행 단위
# 함수 고유의 실행 공간을 갖음
# 자원의 재활용
# 함수 지향적인 언어, 멀티패러다임 지원 = 유연한 언어를 기능하는게 funcion때문에 가능

## 내장함수 일부 체험
print(sum([1,2,3])) #list, set tuple
print(bin(8)) #2진수
print(eval('4 + 5')) # 문자열을 계산하는 수식

### 근사치 값
print(round(1.2), round(1.6))
import math # round는 많이 사용되서 이미 내장되어 있음
print(math.ceil(1.2),math.ceil(1.6)) # 근사치 큰수
print(math.floor(1.2),math.floor(1.6)) # 근사치 작은수


b_list = [True, 1, False]
print(all(b_list)) # 모두 True여야 True
print(any(b_list)) # 하나라도 True가 있으면 True

### 그룹 만들기
data1 = [10, 20, 30]
data2 = ['a', 'b']
for i in zip(data1 , data2):
    print(i)

# 이미 만들어 진거라 있는지 없는지 찾아보고 있으면 사용.