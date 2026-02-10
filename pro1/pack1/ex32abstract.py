'''
추상클래스(abstract class)
    - 추상 메소드를 가진 클래스를 추상클래스라고 함
    - 인스턴스 할 수 없다, 객체 생성 불가, 인스턴스가 안되기 때문에 내용을 적지 X
    - 부모 클래스로만 사용됨
    - 하위클래스에게 오버라이딩 강요
    - method 위에 장식자 @abstractmethod 
'''
from abc import *


class AbstractClass(metaclass=ABCMeta):  #추상클래스, 오버라이딩 필수

    @abstractmethod             # 장식자로 메소드 고정
    def abcMethod(self):        # 추상메소드
        pass                    # 인스턴스가 안되기 때문에 내용을 적지 X


    def normalMethod(self):  #일반메소드, 오버라이딩 선택
        print('추상클래스 내의 일반 메소드')

# parent = AbstractClass()    #에러:추상클래스는 객체 생성 불가(Can't instantiate abstract class)

class child1(AbstractClass):
    name = '난 Child1'

    # 추상메서드
    def abcMethod(self):
        print('부모가 가진 abcMethod 재정의 : 강요당함 ㅠㅠ')

c1 = child1()
print('name :', c1.name)
c1.abcMethod()
c1.normalMethod()

print('-'*100)
class child2(AbstractClass):
    # 추상메서드
    def abcMethod(self):
        print('추상클래스 내의 abcMethod 재정의')

    def normalMethod(self): # 일반메소드 재정의(오버라이딩)
        print('일반메소드 내 맘대로 내용 변경')

c2 = child2()
c2.abcMethod()
c2.normalMethod()

# 다형성.
print('-'*100)
happy = c1
happy.abcMethod()
happy = c2
happy.abcMethod()

