# 함수 장식자
# 기존 함수 코드를 고치지 않고 함수의 앞/뒤 동작을 추가하기
# 함수를 받아서 기능을 덧붙인 새함수로 바꿔치기하는 것
# meta(정보를 가지고 있다) 기능이 있다.

### 장식자 없이 실행
def make2(fn):
    return lambda:"안녕 " + fn() #안녕  반가워 홍길동
def make1(fn):
    return lambda:'반가워 ' + fn() # -> 반가워 홍길동 를 들고 위로 ↑
def hello():
    return '홍길동'
hi = make2(make1(hello)) # hi = 함수의 주소를 가지고 있어.
print(hi()) # 안녕 반가워 홍길동

### 장식자로 실행
print()
@make2 # make1은 make2에게 던져짐
@make1 # hello2가 make1에게 던져짐  ↑
def hello2():
    return '신기해'
print(hello2())

print("---------"*10)

def traceFunc(func):
    def wrapperFunc(a, b):
        r = func(a, b)
        print(f'함수명:{func.__name__}(a={a}, b={b} -> {r})') # __name__ : 현재함수의 이름을 가지고 있는 파일을 리턴함
        return r #함수 반환값을 반환
    return wrapperFunc # 내부함수 주소를 반환(closer)

@traceFunc
def addFunc(a, b):
    return a + b

print(addFunc(10,20))