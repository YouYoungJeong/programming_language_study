# 문1) 1 ~ 100 사이의 정수 중 3의 배수이나 2의 배수가 아닌 수를 출력하고, 합을 출력
num = 0
numsum = 0
while num <=100:
    num += 1
    if (num % 3 == 0):
        if(num % 2 == 1):
            numsum += num
print(numsum)


# 문2) 2 ~ 5 까지의 구구단 출력​
gugu = 1

while gugu <= 5:
    print(f'{gugu}단')
    dan = 1
    while dan < 9:
        dan += 1

        print(f'{gugu} * {dan} = {gugu * dan}',end=" ")
    print()
    gugu += 1

# 문3) 1 ~ 100 사이의 정수 중 “짝수는 더하고, 홀수는 빼서” 최종 결과 출력​
num2 = 1
evennum = 0
oddnum = 0
while num2 <= 100:  
    if num2 % 2 == 0:
        evennum += num2
    else:
        oddnum -= num2
    num2 += 1    
print(f'짝수 = {evennum}, 홀수 = {oddnum}')

# 문4) -1, 3, -5, 7, -9, 11 ~ 99 까지의 모두에 대한 합을 출력​
a = -1
b = 3
sum = 0
while a >-97 and b < 99:
    a -= 4
    b += 4    
    sum += a + b
    # print(sum)
print('문4 = ',sum)

# 문5) 1 ~ 100 사이의 숫자 중 각 자리 수의 합이 10 이상인 수만 출력
# 예) 29 → 2 + 9 = 11 (출력)
print('문5 =')
num3 = 10
while 10 <=num3 <= 100:
    strnum2 = str(num3)
    spritnum2= strnum2[0]+strnum2[1]
    num3 += 1
    intnum = int(strnum2[0])+int(strnum2[1])
    if intnum == 10:
        print(f'{int(strnum2[0])}{int(strnum2[1])}')
    


# 문6) 1부터 시작해서 누적합이 처음으로 1000을 넘는 순간의 숫자와 그때의 합을 출력
# 힌트: 언제 멈출지 미리 모름 → while 적합​
num6 = 1
while num6 <= 1000:
    num6 += num6
print(num6)

# 문7) 구구단을 출력하되 결과가 30을 넘으면 해당 단 중단하고 다음 단으로 이동
print('문7')
gugu7 = 1

while gugu7 <= 9:
    print(f'{gugu7}단')
    dan7 = 1
    while dan7 < 9:
        dan7 += 1
        if gugu7 * dan7 >= 30: continue
        print(f'{gugu7} * {dan7} = {gugu7 * dan7}',end=" ")
    print()
    gugu7 += 1


# 문8) 1 ~ 1000 사이의 소수(1보다 크며 1과 자신의 수 이외에는 나눌 수 없는 수)와 그 갯수를 출력
# num8 = 1
# a= 0

# while num8 <= 1000:
#     num8 += 1
#     if num8
#     print(num8)

# 힌트: 이 문제는 반복이 두 단계다. 2부터 1000까지 하나씩 검사한다. 각 숫자마다 소수인지 확인한다.
# 그래서 while 안에 while 구조가 필요하다.


# continue 연습 문제

# 문제1) 1부터 50까지의 숫자 중 3의 배수는 건너뛰고 나머지 수만 출력하라



# 문제2) 1부터 100까지 출력하되 4의 배수, 6의 배수는 건너뛴다. 그 외의 수 중 5의 배수만 출력하고 그들의 합도 출력하라