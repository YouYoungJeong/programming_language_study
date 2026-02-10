# 여러개의 부품 객체를 조립해서 완성차 생성
# 클래스의 포함관계 사용 (OOP의 핵심 자원의 재활용)
# 다른 클래스(객체)를 마치 자신의 멤버 처럼 선언하고 사용
# 클래스 / 객체=개체=object / 인스턴스 : 구분해서 설명할 줄 알아야함. 책 참고
# import ex23class_has_a_handle
from ex23class_has_a_handle import PohamHandle

class PohamCar:
        turnShowMessage = '정지'

        def __init__(self, ownerName):
                # ownerName = ownerName # X
                # # (import한) 포함핸들클래스의 주소를 가지고 있음 
                # PohamCar는 PohamHandle을 통해 핸들의 정보를 가지고 있음
                
                self.ownerName = ownerName             
                self.handle = PohamHandle() # 클래스의 포함관계           

        def turnHandel(self, q):
                # 핸들에 대해 알고싶으면 car에서 물어보면 X, car는 핸들의 주소만 가지고 있음.
                # 핸들에게 물어봐야함
                if q > 0:
                        self.turnShowMessage = self.handle.rightTurn(q)
                elif q < 0:
                        self.turnShowMessage = self.handle.laftTurn(q)
                elif q == 0:
                        self.turnShowMessage = '직진'

# option 쓰는게 좋다.
if __name__ == '__main__':
        tom = PohamCar('미스터 톰')
        tom.turnHandel(10)
        print(tom.ownerName + "의 회전량은 " + \
        tom.turnShowMessage + ' ' + str(tom.handle.quantity))

        john = PohamCar('존')
        john.turnHandel(-20)
        print(john.ownerName + "의 회전량은 " + \
        john.turnShowMessage + ' ' + str(john.handle.quantity))

        john.turnHandel(0)
        print(john.ownerName + "의 회전량은 " + \
        john.turnShowMessage + ' 0' )


