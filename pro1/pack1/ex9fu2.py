## 사용자 정의
# Class는 대문자 시작, Function은 소문자 시작 추천 
'''
def 함수명(가인수,,,,):
    return 반환값 # 1개만 반환, return이 없으면 return None

함수명(실인수)  # 함수 호출
'''

## 인수X, returnX
def doFunc1():
    print('doFunc1 수행')

doFunc1() # doFunc1을 수행하시오 : 함수호출
print('함수의 주소는 : ', doFunc1) # <function doFunc1 at 0x000001EA19FAF920> 함수명을 적으면 객체의 주소를 찍음 #hack
print('함수의 주소는 : ', id(doFunc1)) #hash code
imsi = doFunc1 # 함수의 주소를 치환
imsi() 
imsi2 = doFunc1() # 함수의 결과를 치환
imsi2
print(doFunc1()) # doFunc1 수행 \n None(돌아온 값 = 돌아 올때 들고 올게 없으면 none를 들고옴)
print('----'*10)

## 인수1개, returnX
def doFunc2(name):
        print('name : ', name)
        # return None -> 반환되는 값이 없으면 써도 되고 안써도 됨. 값이 있으면 무조건 써야함.

## 변수에 값을 주는 방법 a=7 or def(a)<- paramater
# doFunc2() # 애러
doFunc2(7)
doFunc2('길동') # 들어오는 값에대해 알아서 Type을 결정해줌.
# doFunc2('길동','순신') # 인수값이 설정한 갯수보다 많아서 애러

## 인수여러개, returnO
def doFunc3(arg1, arg2):
    re = arg1 + arg2
    return re
print('----'*10)

doFunc3('대한','민국') # 실행만 됨. 찍히지X
print(doFunc3('대한','민국'))
print(doFunc3(5, 6)) # int로 받음.
result = doFunc3('5','6') #str로 받음
print(result)


## 인수여러개, return 응용
print('----'*10)
def doFunc3_1(arg1, arg2):
    re = arg1 + arg2
    if re % 2 == 1:
        return
    else:
        return re
print(doFunc3_1(5, 6))

print('----'*10)
def doFunc4(a1, a2):
    imsi = a1 + a2
    if imsi % 2 ==1:
        return
    else:
        return imsi
print(doFunc4(3,4)) # None
print(doFunc4(3,5)) # 8

## 함수내에서 다른함수 호출 : 함수는 자기를 부른 값으로 무조건 돌아감
print('----'*10)
def triArea (a, b):
    c = a * b / 2
    triAreaPrint(c)

def triAreaPrint(cc):
    print("삼각형의 면적은 : ", cc)

triArea(2,6)

### 반환값 bool
print('----'*10)
def passResult(kor, eng):
    ss = kor + eng    
    if ss >= 50:
        return True
    else:
        return False 
    
if passResult(20, 50):
    print("합격")
else:
    print('불합격')

### 숫자 바꾸기
print('----'*10)
def swapFumc(a, b):
    return b, a     # return (b, a) -> tuple의 형태로 넘어감, 묶음형 자료 하나로 넘어감. 함수의 반환값은 하나만 넘어감.

a = 10; b= 20
print(a, ' ' , b)
print(swapFumc(a, b))


### 내부함수 - 함수 안에서 함수를 선언, 함수를 선언한 함수에서만 사용 가능
print('----'*10)
def funcTest():
    print('funcTest 멤버 처리')
    def funInner():             # funcTest에서만 사용하는 함수  
        print('내부 함수')

funcTest()


### if 조건식 안에 함수 사용
def isOdd(para):
    return para % 2 == 1  # 홀수이면 True반환

mydict = {x:x*x for x in range(11) if isOdd(x)} # true아님 false였을때 숫자를 받아서 dict로 들어와
print(mydict)

## 변수의 생존 범위
print('변수의 생존 범위(scope rule)------')
#변수가 저장되는 이름 공간은 변수가 어디서 선언 되어있는가에 따라 생존시간이 다름
# 전역, 지역 변수

### 전역 변수, 현재 모듈 어디서든 호출 가능, 파일 수준(global)
player = '전국대표'
name = '한국인'

### 지역변수, 현재 함수 내에서만 호출 가능
def funcSoccer():
    name = '홍길동'
    player = '지역대표'
    city = '서울'
    print(f'이름은 : {name} , 수준은 : {player}, 지역은 : {city}')

funcSoccer()

print(f'이름은 = {name} , 수준은 = {player}')

'''
파일.py <- 모듈
안에서 선언하는 변수 = 전역변수
안에서 함수가 선언하는 변수 = 별도의 실행공간 , 함수 영역내에서만 의미가O
하나의 모듈내에서 똑같은 변수 이름 주는건 안좋아. 
'''


def funcSoccer():
    name = '홍길동'
    # player = '지역대표'
    city = '서울'
    print(f'이름은 : {name} , 수준은 : {player}, 지역은 : {city}')
funcSoccer()        # 지역에서 뒤져서 없으면 전역으로 넘어감 -> 전역에도 없으면 애러


print()
a = 10; b = 20; c = 30
def Foo():
    a = 7            # 지역 변수
    b =  100         
    def Bar():   
        global c     # Bar 함수의 멤버가 아니라 모듈(파일)의 멤버가 됨. 전역변수
        nonlocal b   # 한 단계 위 수준으로 올라감(Foo)
        b = 8        # 지역 변수
        print(f'Bar 수행 후 a:{a}, b:{b}, c:{c}')
        c = 9
        b = 200
        
    Bar()
    print(f'Foo 수행 후 a:{a}, b:{b}, c:{c}')
    

Foo()
print(f'함수 수행 후 a:{a}, b:{b}, c:{c}')

print()
g = 1
print('g : ', g)
def func():
    global g   # 실행 오류
    a = g
    g = 2
    return a

print(func())
print('g : ', g)