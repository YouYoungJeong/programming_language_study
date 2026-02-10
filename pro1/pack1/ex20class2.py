class Car:
    handle = 1
    speed = 0

    def __init__(self, name, speed):  #(지역변수)
        self.name = name  #  현재 객체의 name(self.name)에게 지역변수(인자값) name을 치환.
        self.speed = speed
    
    def showData(self):
        km = '킬로미터'
        msg = '속도 : ' + str(self.speed) + km
        return msg
    
    def printHandle(self) :
        return self.handle
    
print(Car.handle)  #  Prototype(원형 클래스의 멤버를 호출), 실무 사용 X
car1 = Car('tom', 10) # 생성자 호출 후 객체생성(인스턴스화 했다.), 실무 사용
car2 = Car('john', 20)
print('car1 객체 주소 : ', car1, '\ncar2 객체 주소 : ', car2)
print('car1 : ', car1.name,' ',car1.speed)
print('car2 : ', car2.name,' ',car2.speed)

# 인스턴스한 객체 변수 추가
car1.color = '파랑'
print(f'car1.color : {car1.color}')
#객체 멤버 확인
print(car1.__dict__)    # {'name': 'tom', 'speed': 10, 'color': '파랑'}
print(car2.__dict__)    # {'name': 'john', 'speed': 20}
# print(Car.color) #AttributeError  : 'Car' object has no attribute 'color'
# print(car2.color) # AttributeError  : 'Car' object has no attribute 'color'

#3개의 객체가 메모리를 따로 가지고(확보) 있어..
print('\n',id(Car), id(car1), id(car2))


'''
UML 다이어그램 : 시스템 모델링을 명시적을 표현하기 위한 모든 언어에 대한 표준 방법
# ** 시퀀스 , 클래스, 유스케이스 다이아그램 꼭 읽을 줄 알아야함!

<전역>
    Car (class): prototype(원형클래스)안에 있는 멤버들, 객체 공간(메모리) , 공유자원
    진짜 “클래스(객체)의 멤버변수”란? -> 조건 : self. 가 붙어야 한다
┌---------------------Car--------------------┐
|   <public(+)- 멤버, object들의 공유자원>
|   handle          변수 
|   speed           변수
|   self.name       인스턴스 멤버 변수
|   self.speed      인스턴스 멤버 변수
|   -----------------------------
|   생성자, 소멸자   특수 메소드(매직 메소드)
|   showData(self)        메소드
|   printHandle(self)     메소드
└--------------------------------------------┘
    Car.handle을 호출하면 handle을 부름

멤버변수 종류 정리
위치	예시	정체
클래스 바로 아래	handle = 1	클래스 멤버 변수
생성자에서 self.	self.speed	인스턴스 멤버 변수
메소드 안 지역	km, msg	지역변수 ❌

<지역>
car1의 인스턴스함 (객체 공간이 만들어짐) -> car1 객체의 주소를 기억 -> self 는 car1의 주소를 가지고 있다.

┌---------car1 : 객체 공간, 공유 X -------┐
    handle = 1      변수 
    name = 'tom'    변수 
    speed = 10      변수
    color = '파랑'  # 장비추가 가능 :       각각의 객체에 변수 추가 가능
└-----------------------------------------┘
        
┌--------car2 : 객체 공간, 공유 X ---------┐
    handle          변수 
    name = 'john'   변수 
    speed = 20      변수
└-----------------------------------------┘
    car1 = Car('tom', 10) 호출할때 self생량되었지만 self가 객체공간의 주소를 가지고 있어
    실제로는 (self=car1의 주소,객체공간).name, (self=car1의 주소,객체공간).speed로들어감
    car1에 handle값이 주어지지 않았다면 prototype(원형클래스,공유멤버)으로 돌아가서 호출.
'''
# 메소드 는 공유 self를 통해 각각의 객체를 구분.
print('-'*100)
print(f'car1 speed : {car1.showData()}') # = car1.showData(self) (car1)<-인터프리터가 알아서 class 안의 메서드로 들고감 ,넣으면 에러
print(f'car2 speed : {car2.showData()}')
car1.speed = 80
car2.speed = 110


print(f'car1 handle : {car1.printHandle()}') # car1의 handle값을 찾고 없으면 원형클래스 값을 찾는다 : 바로 원형클래스 값을 찾지 않음.
print(f'car2 handle : {car2.printHandle()}')