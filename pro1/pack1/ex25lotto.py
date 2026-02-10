#클래스의 포함 관계 연습1 - 로또 번호.(append 사용)
import random

class LottoBall: # 각각의 객체에 각각 등록해야해서 멤버필드 생성X , 번호 공유를 안하니까
    def __init__(self, num):
        self.num = num


class LottoMachine:

    # 초기에 볼이 들어있는 상태
    def __init__(self):
        self.ballList = [] 
        for i in range(1, 46):
            self.ballList.append(LottoBall(i)) # <- 클래스의 포함관계

    # 볼 뽑기
    def selectBalls(self):
        # for a in range(45): # list값 확인하기(0~45)
        #     print(self.ballList[a].num, end=" ")

        # print('\n','-'*100)    
        random.shuffle(self.ballList) # 번호섞기
        
        # for a in range(45): # list값 확인하기(0~45)
        #     print(self.ballList[a].num, end=" ")
        return self.ballList[0:6]


class LottoUI:
    def __init__(self):
        self.machine = LottoMachine() # 포함관계
    
    # 버튼을 눌러 볼이 나옴
    def playLotto(self):
        input('로또를 시작하려면 엔터키를 누르세요')
        selectedBall = self.machine.selectBalls() # LottUI는 selectBalls을 보려면 machine을 통해 볼 수 있다.
        for ball in selectedBall:
            print('%d'%(ball.num))


if __name__ == '__main__':
    # machine = LottoMachine()
    # print(machine.selectBalls())

    # 객체 변수로 호출
    # lot = LottoUI()
    # lot.playLotto()

    # 객체 만들어서 호출 : 한번만 부르고 말거면 사용
    LottoUI().playLotto()