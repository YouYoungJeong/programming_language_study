# 조건 판단문 if 

## if
var = 3

if var >= 3:
    print('크네') #조건이 True면 빠져나와
    
print('end')

var = 1

if var >= 3:
    print('크네') #조건이 false면 skip

print('end')

if var >= 3:
    print('크네') #들여쓰기로 집단을 구분함.
print('흠 느낌')
print('end')

if var >= 3:
    print('크네') #들여쓰기로 집단을 구분함.
    print('흠 느낌')
print('end')

## if else
var = 3
if var >= 3:
    print('크네') #들여쓰기로 집단을 구분함.
else:
    print('작구나')
print('end')

var = 2
if var >= 3:
    print('크네') #들여쓰기로 집단을 구분함.
else:
    print('작구나')
print('end')


print('-----'*10)
money = 1000
age = 25
if money >=500:
    item = '사과'
    if age <= 30:
        msg = '참 참'
    else:
        msg = '참 거짓'
else:
    item = '한라봉'
    if age >= 20:
        msg = '거짓 참'
    else:
        msg = '거짓 거짓'
print(f'중복 if 수행후 결과 {item} {msg}')
print('end')

print('-----'*10)
money = 200
age = 35
if money >=500:
    item = '사과'
    if age <= 30:
        msg = '참 참'
    else:
        msg = '참 거짓'
else:
    item = '한라봉'
    if age >= 20:
        msg = '거짓 참'
    else:
        msg = '거짓 거짓'
print(f'중복 if 수행후 결과 {item} {msg}')
print('end')

print('-----'*10)

## elif
jumsu = 85
if jumsu>= 90:
    print('우수')
elif jumsu >=80:
    print('보통')
else:
    print('저조')
print('end')
print('-----'*10)

## input : 입력값은 모두 문자열 타입
# data = input('점수:')
# jumsu = int(data)
if jumsu>= 90:
    print('우수')
elif jumsu >=80:
    print('보통')
else:
    print('저조')
print('end')

print('-----'*10)
## elif
jumsu = 85
if jumsu>= 90:
    print('우수')
elif jumsu >=80:
    print('보통')
else:
    print('저조')
print('end')

print('-----'*10)
# Python 에만 있는 조건
jum = 80
if 90 <= jum <= 100:
    print('A')
elif 70 <= jum <= 90:
    print("B")
else:
    print('C')
print('end')

## 대입 표현식
print('-----'*10)
names = ['홍길동',' 신선해','이기자']
if '홍길동' in names:
    print('친구 이름이야')
else:
    print('누구야?')


if (count := len(names)) >=5: # 대입 표현식 괄호가 먼저 적용 count <- 변수 이름
    print(f"인원수가{count}명 이므로 단체 할인 적용")
else:
    print('ㅠㅠ')
scores = [95, 88, 76, 81]
if (avg :=sum(scores) / len(scores)) >= 80: # avg<-변수이름
    print(f"우수반 : 평균점수{avg}")

## 삼항연산 (다중 if가능하지만 if는 하나만 쓰는것을 권장)
print('삼항 연산')
a = 'kbs'
b = 9 if a == 'kbs' else 11
print('b :', b)


a=11
b = 'mbc' if a == 9 else 'kbs'
print('b :', b)

a = 3
if a < 5:
    print(0)
elif a < 10:
    print(1)
else:
    print(2)
# 위의 문장을 삼항연산인 한줄로 표현하면
print(0 if a < 5 else 1 if a < 10 else 2) # 권장하지 X
print('end')