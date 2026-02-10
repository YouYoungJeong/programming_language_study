'''
클래스 강결합 - 상속 <- 유
지 보수가 어렵다, 실업무에서는 자주 사용 X
                    <-  다형성을 구사하려면 무조건 상속 사용(매력적)
                    UML : -▷
클래스 약결합 - 포함 <- 탈부착이 쉽다. 사용 권장.

상속 :
- 자원의 재활용을 목적으로 특정 클래스의 멤버(속성과:멤버필드, 행위:메소드)를 가져다 쓰는 것
- 코드 재사용
- 확장성 - 기존 클래스에 새 기능을 추가한 새로운 클래스 생성
- 구조적 설계 - 공통개념은 부모 클래스, 구체적 내용은 자식 클래스에서 구현
- 다형성 구사 - 메소드 *오버라이딩


'''
# 추상적, 부모 클래스는 따로 모듈로 만드는걸 추천
class Animal:                   #동물들이 가져야할 공통 속성과 행위 선언

    age = 1

    def __init__(self):
        print('Animal 생성자')

    def move(self):
        print('움직이는 생물')


'''
class Dog(Animal) = 자식클래스(부모클래스)
- Animal - 부모, 조상, super, parent 클래스라 불림.
- Dog - 자식, 자손, 파생, sub , child 클래스라 불림

'''

# 구체적
class Dog(Animal): # 상속 
    
    def __init__(self):
        print('Dog 생성자')
        # ani = Animal() # 포함관계 -> Animal의 생성자 호출됨.
    
    def my(self):
        print("댕댕이라고 해요")


# 인스턴스 하기(인스턴스 베리어블, 오브젝트 베리어블)
dog1 = Dog()


# 상속함으로 인해 Dog()에서 Aminal의 값을 가져 올 수 있다.
dog1.my()
dog1.move()
print('age : ', dog1.age)

print('-'*100)
dog2 = Dog()
dog2.my()
dog2.move()

# 구체적
print('-'*100)
class Horse(Animal): # 상속
    pass
horse1 = Horse() 
horse1.move()