# 재귀(Recursion) : 함수 자신이 자기를 자꾸 부름. 반복적으로 해결해야 하는 방식으론 for, while이랑 같다. 동작방식, 메모리사용, 사용목적차이가 존재
# 재귀 함수는 메모리 사용에 대해 무조건 알아야함 : 메모리 소모가 많아짐. 함수는 별도의 공간을 가지고 있기때문에 그 공간을 가지고 반복하기 때문
# -> 순수 반복은 for나 while를 사용
#https://cafe.daum.net/flowlife/RUrO/157 정리
# 재귀함수 : 함수가 자기 자신을 호출 - 반복처리
# 알고리즘 할때 재귀가

def countDown(n):
    if n == 0:
        print('완료')
        return # 빠져나오는걸 명시적으로 표시 안적어도 됨. 적어야 할 때가 있음.
    else:
        print(n, end=' ')
        countDown(n - 1) # 재귀 호출
countDown(5)

print('-'*100)
# 1부터 n 까지 합
def totFunc(n):
    if n == 0:
        print('exit')
        return 1 # 1을 반환
    return n + totFunc(n-1) # 재귀함수
    

result = totFunc(5)
print('result : ', result)

'''
5를 가지고 들어가서 tot(5)에서 호출 
->  5+tot(4)                         10+5
->  tot(3)                           6+4
->  tot(2)                           3+3
->  tot(1)                           2+1
->  탈출                   return 값 : 1
->  return                           여기서 위로 올라가면서 연산 
    호출할때는 연산 X. 호출 완료 되면     ^^^
'''



print('-'*100)
# 계승 (factorial)
def factFunc(a):
    if a == 1 :  return 1
    print(a)
    return a * factFunc(a - 1)

result2 = factFunc(5)
print('result2 : ', result2)

'''
5를 가지고 들어가서 tot(5)에서 호출 -> 
    5*fact(4)                            24*5
->  4*fact(4)                            6*4
->  3*fact(4)                            2*3
->  2*fact(4)                            2*1
->  탈출                        return 값 : 1
->  return                           여기서 위로 올라가면서 연산 
호출할때는 연산 X. 호출 완료 되면     ^^^
'''

print('end')