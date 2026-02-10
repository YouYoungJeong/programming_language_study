# oop(객체 중심 프로그램) : 새로운 타입 생성, 포함, 상속, 다형성 등을 구사
# 보일러플레이트
# 함수만으론 opp를 할 수 없어.
# Class(설계도)로 인스턴스 해서 객체를 생성(별도의 이름 공간을 갖음), 클래스를 이용해야 다양성을 구성 할 수 있다.
# 객체는 멤버필드(멤버변수) 와 메소드로 구성
# (java랑 다르게) 접근 지정자가 없다(퍼플릭, 프로테틱). 메소드 오버로딩 없다.
# 모듈의 멤버 : 변수, 명령문, 함수, 모듈, 클래스


print('뭔가를 하다가 객체 만들기')

# class 선언
# 새로운 타입을 만들기 위한 설계도 만들기
class TestClass: # 부모없으면 그냥 :
    aa = 1 # 멤버필드(변수) : '현재 클래스 영역 내'에서 전역변수, 없어도 됨.(속성)

    def __init__(self):     # 특별한 메소드, 초기화 할게 없으면 안 적어도됨(내부적으로 인터프리터가 만들어줌), 생성자는 하나밖에 못씀.
        print('Class의 메소드 이름이 __init__ 이면 생성자-> 객체 생성시 가장 먼저 1회만 호출 - 초기화 담당')

    def __del__(self):
        print('소멸자 : 프로그램 종료시 자동실행, 마무리 작업.')   #프로그램이 끝나면 시스템에 의해 자동으로 부름(callback), 
                                                                #마무리 할게 없으면 안 적어도됨, 잘 안씀

    def printMsg(self):         # class의 "행위"를 하는 부분은 메소드라 부름,일반 메소드 : 메소드는 반드시 argument 를 가져야함 그게 self
        name = '한국인'         # 지역변수 : printMag 에서만 유효
        print(name)

print(TestClass)    # <class '__main__.TestClass'> -> type이 TestClass야
test = TestClass()  # 객체변수를 한개 생성, 인스턴스 한다 (객체 = 클래스)
# 클래스 멤버 확인 
print('test객체의 멤버 aa : ',test.aa)

#메소드 호출하기(method call)
## 호출방법 1 . Bound Method call- 객체변수를 이용해서 호출 했기 때문에 알아서 객체변수를 넣어서 불러
test.printMsg()   
## 호출방법 1 . Unbound Method call 
# TestClass.printMsg() # argument에러 - 클래스의 이름으로 불렀기 때문에 앞에 만들어진 객체변수를 넣어서 불러야함(그래야 주소를 알아.)
TestClass.printMsg(test)

print(type(1))
print(type(1.0))
print(type(test))
print(id(test))         # 설계도에 의해 객체가 만들어짐 
print(id(TestClass))    # 설계도의 생성자에 의해 객체가 만들어짐

test2 = TestClass() #  객체변수를 한개더 생성
print(id(test2))

