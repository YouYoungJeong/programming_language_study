''' 
메소드 오버라이딩 ( = 재정의)
    - 부모에서 정의된 메소드의 자식이 동일명의 메소드로 내용만 변경해 사용하는것.
    - 부모 고유의 기능을 대체하는 새로운 기능
    - *동작의 구체화(공통 틀은 부모가, 실제 행동은 자식) 실현
    - Polymorphism(다형성) : 같은 메소드나, 객체에 따라 다른 기능을 수행
    - 확장, 유지보수에 도움 : 부모 코드 유지한채 자식 코드만 변경
'''

class Parent:
    # 자식들이 가지고 갔으면 좋은 메소드 생성
    def printData(self):
        pass


class Child1(Parent):

    def abc():
        print('Child1 고유 메소드')

    def printData(self):    # 매소드 오버라이딩
        a  = 5 + 6
        # ...
        print('Child1에서 printData override') 


class Child2(Parent):

    def printData(self):    # 매소드 오버라이딩
        print('Child2에서 printData 재정의')
        msg = '부모와 동일한 메소드명이나 내용은 다르다'
        print(msg)

# 오버라이딩
c1 = Child1()
c1.printData()
print('\n')
c2 = Child2()
c2.printData()

# 다형성 : 문장(statment)이 같은데 출력내용이 달라.
# 추천
print('\n다형성 ----')
par = Parent()

par = c1
par.printData() # Child1의 메서드 출력
print()

par = c2
par.printData() # Child2의 메서드 출력
print()

print('-'*100)
# java 스타일의 다형성. 자바는 부모에게 직접 주지 않아, 비추 다형성을 추측하기가 어려워서
imsi = c1
imsi.printData()
imsi = c2
imsi.printData()