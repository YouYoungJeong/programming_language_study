"""
알고리즘(Algorithm)
컴퓨터가 문제를 해결하거나 특정 작업을 수행하기 위해 따라야 하는 
단계적인 절차, 규칙 또는 명령어들의 집합
입력 값을 통해 원하는 결과(출력)를 얻기 위해 논리적으로 정돈된 순서를 따르며
일상생활의 요리법이나 칫솔질 순서도 알고리즘의 예시.

빅오 표기법 (big-O notation)
알고리즘의 공간복잡도(메모리) - 적은 메모리를 사용하여 : 요즘은 잘 신경 X
<코테에서 가장 많이 신경쓰는부분>
알고리즘의 시간복잡도(계산) 
    - 하나의 문제에 소요시간값이 얼마나 짧은시간 안에 나오는지.
    - (fast) 상수함수 < 로그함수 < 선형함수 < 다항함수 < 지수함수 (slow)
"""

print("-"*40," 1부터 n까지 연속한 숫자의 합을 구하는 알고리즘 ","-"*40)
# 1부터 n까지 연속한 숫자의 합을 구하는 알고리즘.
# 빅오 표기법 : O(n)
print('==for문 사용==')
def sum_n(n):
    s = 0
    for i in range(1, n+1):
        s += i
    return s
print(sum_n(10))
print(sum_n(100))

# 디버깅, 코드를 하나하나 한줄씩 실행할때 행 옆에 점 찍고 좌측 네비바 실행및 디버그 실행
# 한단계 한단계 확일 할 수 있다.

# 가우스 덧셈 공식 n까지의 합 (첫 번째 수 + 마지막 수) * 항의 개수 / 2
# 숫자값이 아무리 커져도 연산은 딱 3번만 하기 때문에 연산이 빠르다.
# 빅오 표기법 : O(1)
print('==가우스의 합공식 사용==')
def sum_n2(n):
    return n * (n+1) // 2
print(sum_n2(10))
print(sum_n2(100))

print("-"*40," 최대값 구하기 알고리즘 ","-"*40)
# O(n) : for 문
d = [17, 92, 33, 58, 7, 32, 42]

print('==for문 사용==')
def find_max(a):
    # 입력 크기 갯수를n이 가짐
    n = len(a)
    # 첫번째 값 자리 주기
    maxv = a[0]
    for i in range(1, n):
        if a[i] > maxv:
            maxv = a[i]
        return maxv
print(find_max(d))

print("-"*40," 두 수 의 최대 공약수 구하기 ","-"*40)
# 두 수 의 최대 공약수 구하기
# 예) 4 , 6 : 4와 6은 모두 2로 나누어 떨어지므로 2가 최대공약수(GCD) 
# 아래 방법은 빠를 수도 느릴 수도 있다. 
print('==while문 사용==')
def gcdFunc(a, b):
    i = min(a, b)
    while True:
        if a % i == 0 and b % i == 0:
            return i
        i -= 1

print(gcdFunc(4, 6))
print(gcdFunc(24, 16))
print(gcdFunc(81, 27))

# 유클리드 알고리즘(호제법)
# 두 자연수 a, b (a > b)의 최대공약수(GCD)를 효율적으로 구하는 방법으로, 
# a를 b로 나눈 나머지 r이 0이 될 때까지 b 와 r의 GCD를 반복해서 구하는 방식
print('==유클리드 알고리즘 사용 ( 재귀 함수 )==')

# 두수중 b가 0이면 앞에수 a를 리턴
def gcdFunc2(a, b):
    if b == 0:
        return a
    # b와 a % b 나머지 연산 -> 좀더 작은값으로 재귀 호출
    return gcdFunc2(b, a % b)

print(gcdFunc2(4, 6))
print(gcdFunc2(24, 16))
print(gcdFunc2(81, 27))