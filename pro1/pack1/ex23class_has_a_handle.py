'''
OOP
성격이 다른 각각의 object를 불러서 쓰는게 객체 지향
조립식 프로그램
-> class를 알아야 함
has a : class 의 포함 관계 각각의 오브젝트를 멤버로 사용
    - 자동차 예시
is a  : class 의 *상속* 관계, 각각의 오브젝트를 부모로 사용
'''
# 그림 그리기!
# 회전이 필요한 어딘가에서 필요한 부품 핸들 클래스 작성
class PohamHandle:
    quantity = 0 # 핸들 회전량, PohamHandle의 객체들의 공유자원

    def laftTurn(self, quentity): # laftTurn의 quentity 지역변수
        self.quantity = quentity
        return '좌회전'
    
    def rightTurn(self, quentity):
        self.quantity = quentity
        return '우회전'
