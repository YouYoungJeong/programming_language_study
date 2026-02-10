# Class 부연 설명

# 변수, 필드

kor = 100 # 모듈의 전역변수

# Function
def abc():
    kor = 0 # 함수 내의 지역변수
    print('모듈의 멤버 함수')

class My:
    kor = 80 # My 멤버 변수(필드)

    # Method
    def abc(self):
        print('My 멤버 메소드')

    def show(self):
        # kor = 77      # 메소드 내의 지역 변수
        print(kor)      #77 지역변수 찾고  / 100 없으면 모듈의 멤버를 찾아감 
        print(self.kor) # 80 class 의 멤버 필드
        self.abc()      # class의 self가 들어감
        abc()

my = My()
my.show()
print(my.abc()) # None 

# ↓마무리 확인(정리 생략가능)--------------
print('-'*100)
print(My.kor)

tom = My()
print(tom.kor)
tom.kor = 88
print(tom.kor)

oscar = My()
print(oscar.kor)
# ↑----------------------------------