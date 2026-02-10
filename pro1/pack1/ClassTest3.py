# 다중상속 연습문제

# super
class Animal:

    def move(self):
        print('대부분의 동물들은 4발로 걸어요')
        pass

class Dog(Animal):
    
    # Field
    name = '개'

    # Method
    def move(self):
        print('댕댕이')


class Cat(Animal):
    
    # Field
    name = '고양이'

    # Method
    def move(self):
        print('냥냥이')


class Wolf(Dog, Cat):
    pass


class Fox(Cat, Dog):
    
    #Method
    def move(self):
        print('여우')
    
    def foxMethod(self):
        # pass : 자식이 있나보다 추측함 고유메서드는 pass쓰지마~
        print('fox의 고유 메서드')

animal = [Dog(), Cat(), Wolf(), Fox()]
for a in animal:
    print('-------'*10)
    print(a)      
    a.move()