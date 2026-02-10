# Overriding2 : 결제 시스템으로 다시

# Super class : 자식클래스에 대세 공통 규칙 선언
class Payment():
    def pay(self, amount):
        print(f'{amount}원 결제 처리')
    #def ....

# Payment의 자식은 결제를 pay()라는 동일 메소드를 이용하기를 기대 - 해도되고 안해도 되고 강제성 X
# 동일 인터페이스 구사
# 강제성을 줄 수도 있다 = 추상

class CardPayment(Payment):
    #얘만의 고유 멤버..
    # ''        메소드가 있음 (생략)
    def pay(self, amount):
        print(f'{amount}원 카드 결제 승인 완료함')

class CashClass(Payment):
    #.....
    def pay(self, amount):
        print(f'{amount}원 현금 결제 완료함 - 감사합니다.')


# list의 요소로 두개의 클래스 객체를 집어 넣음.
payment = [CardPayment(), CashClass()]

for p in payment:  # <--- 다형성: 같은 이름으로 수행해서 내용이 다름. 이름이 같으니까 for문도 가능, oop에서 엄청 많이 사용함.
    p.pay(5000) 
