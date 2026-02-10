# 클래스의 포함 관계 연습1 - 냉장고 객체에 음식 객체 담기.(append 사용)

class Fridge:
    isOpend = False
    foods = []

    def open(self):
        self.isOpend = True
        print("냉장고 문을 열기")

    def close(self):
        self.isOpend = False
        print("냉장고 문을 닫기")
    
    def foodList(self): # 냉장고 문이 열린 경우 음식물 확인
        for f in self.foods:
            print(f' - {f.name} {f.expiry_date}')
        print()

    def put(self, thing):
        if self.isOpend: # == True 는 쓸 필요가 없어 isOpend자체가 True를 반환함
            self.foods.append(thing) # 포함
            print(f'냉장고에 {thing.name} 넣음')
            self.foodList()
        else:
            print("냉장고 문이 닫혀있음")
    


# 모듈로 만드는걸 추천
class FoodData:
    def __init__(self, name, expiry_date):
        self.name = name
        self.expiry_date = expiry_date



fObj = Fridge()
apple = FoodData('사과', '2026-8-1') # 객체변수 - 객체 멤버 확인하려고
cola = FoodData('콜라', '2027-11-1')
# 문이 닫혀있음
fObj.put(apple)

print('-'*100)
# 문을 열고 사과를 넣은다음 문을 닫는다 -> 사과 존재
fObj.open()
fObj.put(apple)
fObj.close()

print('-'*100)
# 문을 열고 콜라를 넣은다음 문을 닫는다 -> 사과, 콜라 존재(append)
fObj.open()
fObj.put(cola)
fObj.close()

cola = FoodData('콜라', '2027-11-1')