# Closure : Socpe에 제약을 받지 않는 변수들을 포함하고 있는 코드블럭이다.
# 내부 함수의 주소를 반환해 함수 밖에서 함수 내의 멤버를 참조하기

def funcTimes(a, b):
    c = a * b
    return c

print(funcTimes(2, 3))
# print(c) : Err

kbs = funcTimes(2, 3)
print(kbs)
kbs = funcTimes
print(kbs)
print(kbs(2, 3))
print(funcTimes(2, 3))
print(id(kbs), id(funcTimes))
mbc = sbs = kbs
del funcTimes    # funcTimes 변수 삭제
# print(funcTimes(2, 3))  # NameError: name 'funcTimes' is not defined
print(mbc(2, 3))

print('\n--클로저를 사용하지 않은 경우-----------')
def out():
    count = 0
    def inn():
        nonlocal count
        count += 1
        return count
    print(inn())

# print(count)  # err
out()
out()

print('\n--클로저를 사용한 경우-----------')
def outer():
    count = 0
    def inner():
        nonlocal count
        count += 1
        return count
    return inner     # <== 클로저 : 내부함수의 주소를 반환

var1 = outer()       # 내부함수의 주소를 변수에 저장
print('var1 주소:', var1)
print(var1())
print(var1())
myvar = var1()
print(myvar)
print()
var2 = outer()    # *새로운 객체(inner 함수) 생성 : 아우터 타입의 객체가 2개 생김.
print(var2())
print(var2())

print(var1, var2) # 두개의 주소가 다르다. 둘다 inner function이지만 count값이 다르다. 별도로 돌아감.


print("수량 * 단가 * 세금한 결과를 출력하는 함수 작성 (closure이용)")  # 세금은 분기마다 달라진다는 조건 외부에서 입력받아서 출력

def outer2(tax): #tax는 지역변수(함수안에서 선언한 parameter도 지역변수)
    def inner2(su, dan):
        amount  = su * dan * tax
        return amount
    return inner2
# 1분기에는 su * dan 에 대한 tax는 0.1 부과
q1 = outer2(0.1) # q1은 inner2의 주소를 기억
result1 = q1(5, 50000)
print("result1 : ", result1)

result2 = q1(2, 10000)
print("result2 : ", result2)
print()
# 2분기에는 su * dan 에 대한 tax는 0.05 부과
q2 = outer2(0.05) # q1은 inner2의 주소를 기억
result3 = q2(5, 50000)
print("result3 : ", result3)
result4 = q2(2, 10000)
print("result4 : ", result4)

print("---------"*10)
## 일급함수
print('일급함수 :  함수 안의 함수, 인자로 함수 전달, 반환값이 함수')
def func1(a, b):
    return a + b
func2 = func1
print(func1(3,4))
print(func2(3,4))
print()
def func3(fu):   # 인자로 함수전달
    def func4(): # 함수 안의 함수 선언
        print('나는 내부함수야~')
    func4()
    return fu   #반환값이 함수
mbc = func3(func1)
print(mbc(3,4))

print("---------"*10)
## 축약함수(Lambda function)
print('축약함수(Lambda function) : 이름이 없는 한 줄짜리 함수, **함수로 매개변수로 사용할 때 많이 사용')
# 형식 :  lambda 매개변수,,,(인자s) : 반환식  <- return 없이 결과 반환
def hapFunc(x,y):
    return x + y
print(hapFunc(1,2))
#람다로 표현
print((lambda x,y: x+y)(1,2))
gg = lambda x,y: x+y
print(gg(1,2))

kbs = lambda a, su=10:a + su #변수 하나만 선언해도됨, 초기값을 미리줘서
print(kbs(5))
print(kbs(5,6))

sbs = lambda a, *tu, **di :print(a, tu, di) #아규먼트 이런게가능.
sbs(1, 2, 3, var1=4, var2 = 5)

li = [lambda a,b:a+b, lambda a,b:a*b] # list의 요소가 람다.
print(li[0](3,4))
print(li[1](3,4))

print("---------"*10)
### filter(function, 반복가능한 객체(iterable) : 내장함수)
print('\n 다른 함수에서 람다 사용하기')
print(list(filter(lambda a: a < 5, range(10)))) # 반복가능한 객체안에 5미만인 값을 list로 출력해줘
print(list(filter(lambda a: a % 2, range(10)))) # % a를2로나눈 나머지 0 아니면 1, 0 = false

### 문 ) 1 ~ 100 사이의 정수 중 5의 배수이거나 7의 배수만 출력(filter사용)
print(list(filter(lambda a : a % 5 == 0 or a % 7 == 0,range(1,101))))

