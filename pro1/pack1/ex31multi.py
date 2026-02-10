'''
class의 다중상속
    - 부모가 복수
    - 순서가 중요

'''
class Tiger:
    data = '호랑이 세계'

    def cry(self):
        print('호랑이 : 어흥')

    def eat(self):
        print('맹수는 고기를 좋아함')


class Lion:

    # Tiger, Lion이 동일한 이름의 메소드를 가지고 있음
    def cry(self):
        print('사자 : 으르렁')

    def hobby(self):
        print('백수의 왕은 낮잠이 취미')


# 다중상속 은 순서가 중요
class Liger1(Tiger, Lion):
    pass

a1 = Liger1()
print(a1.data)
a1.eat()
a1.hobby()
a1.cry()    # 호랑이 : 어흥 <- Tiger가 나옴

print('-'*100)

def hobby():
    print('모듈의 멤버 : 일반 함수')

class Liger2(Lion, Tiger):
    data = '라이거 만세'

    def play(self):
        print('라이거 고유 메소드')

    def hobby(self): # 메소드 오버라이드
        print('라이거는 공원 걷기를 좋아함')
    
    def showData(self):
        self.hobby()        # Liger2 -> Lion -> Tiger
        super().hobby()     # Lion -> Tiger
        hobby()             # module

        self.eat()
        super().eat()

        print(self.data + ' ' + super().data)


a2 = Liger2()
a2.cry()
a2.showData()

