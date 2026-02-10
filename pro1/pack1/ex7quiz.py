# python 기초 연습 문제
# 문1 1 ~ 100 사이의 정수 중 3의 배수이나 2의 배수가 아닌 수를 출력하고, 합을 출력
# i=0
# sum=0
# while i<100:
#    i+=1
#    if i%2==0:continue
#    if i%3==0:continue
#    print(i)
#    sum=sum+i
# else:
#   print(sum)

i = 0
total = 0
while i <= 100:
    if i % 3 == 0 and i % 2 != 0:
        print(i, end = ' ')
        total += i
        i += i
    i += 1

print(f'합은{total}')


#문제2 2 ~ 5 까지의 구구단 출력

i = 2
while i <= 5:
    j = 1
    while j <= 9:
        print(f'{i} * {j} = {i*j}')
        j += 1
    print()
    i += 1

# 문3) 1 ~ 100 사이의 정수 중 “짝수는 더하고, 홀수는 빼서” 최종 결과 출력
total = 0
i = 1
while i <= 100:
    if i % 2 == 0:
        total += i
    else:
        total -= i
    i += 1
print(total)

# 문4) -1, 3, -5, 7, -9, 11 ~ 99 까지의 모두에 대한 합을 출력

total = 0
num = 1
sign = -1

while num <= 99:
    total += sign * num
    sign *= -1
    num += 2

print(total)
#문5) 1 ~ 100 사이의 숫자 중 각 자리 수의 합이 10 이상인 수만 출력


# i=10
# while i<=100:
#    str_i= str(i)
#    hap=sum((int(str_i[0]),int(str_i[1])))
#    if hap>=10:
#        print(f'{str_i}일 때 {hap}')
#    i+=1

num = 1
while num <= 100:
    temp = num
    digit_sum = 0

    while temp > 0:
        digit_sum += temp % 10
        temp //= 10
    if digit_sum >= 10:
        print(num, end = ' ')


#문6) 1부터 시작해서 누적합이 처음으로 1000을 넘는 순간의 숫자와 그때의 합을 출력
num = 1
total = 0

while total <= 1000:
    total += num
    num += 1

print('숫자', num - 1)
print('합', total)
# 문7) 구구단을 출력하되 결과가 30을 넘으면 해당 단 중단하고 다음 단으로 이동
for i in range(1,10):
    for j in range(1,10):
        if i*j > 30:
            break
        print(f"{i}X{j}={i*j}")
    print()

dan = 2
while dan <= 9:
    i + 1
    while i <= 9:
        result = dan * i
        if result > 30:
            break
        print(dan, "*", i, "=", result)
        i += 1
        print()
        dan += 1



#문8) 1 ~ 1000 사이의 소수(1보다 크며 1과 자신의 수 이외에는 나눌 수 없는 수)와 그 갯수를 출력
count = 0
for i in range(2,1001):
    flag = 0
    for j in range(2,i-1):
        if i%j == 0:
            flag = 1
            break
    if flag == 0:
        print(i)
        count += 1
print(count)

num = 2
count = 0
while num <= 1000:
    i = 2
    is_prime = True
    while i < num:
        if num % i == 0:
            is_prime = False
            break
        i += 1
    if is_prime:
        print(num, end = ' ')
        count += 1
    num += 1

print('갯수 : ', count)
#문제9) 1부터 50까지의 숫자 중 3의 배수는 건너뛰고 나머지 수만 출력하라
for i in range(1,51):
    if i%3 == 0:
        continue
    print(i, end= '')


i = 1
while i <= 50:
    if i % 3 == 0:
        i += 1
        continue
    print(i, end=' ')
    i += 1


# ​문제 10) 1부터 100까지 출력하되 4의 배수, 6의 배수는 건너뛴다. 그 외의 수 중 5의 배수만 출력하고 그들의 합도 출력하라

s = 0
for i in range(1,101):
    if i%4 == 0 or i%6 == 0:
        continue
    if i%5 == 0:
        print(i)
        s += i
print(s)

print()
i = 1
total = 0
while i <= 100:
    if i % 4 == 0 or i % 6 == 0 or i % 5 != 0:
        i += 1
        continue
    print(i)
    total += i
    i += 1

print('total : ', total)


