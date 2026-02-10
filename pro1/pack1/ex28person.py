# 상속
# 부모 클래스 - 모듈로 만드는거 추천
class Person:
    
    # Member Filed
    say = '난 사람이야~ '
    nai = '20'

    '''
        public 멤버
            - python은 기본이 public임.

        private멤버
            - 생성 하는법 : __변수
            - 다른 클래스에서 참조 할 수 X 
            - Person class에서만 호출 가능
    '''
    __msg = 'good : private멤버'


    #method
    def __init__(self, nai):
        print('Person 생성자')
        self.nai = nai # <- 이해하기!

    def printInfo(self):
        print(f'나이:{self.nai}, 이야기:{self.say}') 
        '''
        - self.say 현재 person클래스의 객체 주소, 
        - self가 없으면 현재 함수의 지역변수 -> 매소드의 전역변수로 감
        - self.nai, self.say는 Person 클래스가 아니라 Person으로 만든 객체(인스턴스)의 멤버필드
        '''

    def helloMethod(self):
        print('안녕')
        print('hello : ', self.say, self.nai, self.__msg)


#부모클래스로 호출------------------------------------------------------------
#바로 출력(이미 객체(멤버)가 설정되어 있어서 가능)
print(Person.say, Person.nai)

# 아직 인스턴스 생성을 안했기 때문에 에러! 
#(TypeError: Person.printInfo() missing 1 required positional argument: 'self')
# Person.printInfo()

# 인스턴스 생성
per = Person('25')
per.printInfo()
per.helloMethod()
#------------------------------------------------------------------------------

# 자식 클래스 생성
print('='*100)
class Employee(Person): # 상속(Employee의 살림살이가 늘었어)

    # Member Field
    subject = '근로자'
    '''
    부모에게 있는 멤버필드 생성해보기 
    - local이 우선
    - 자식이 선언하는 순간 부모는 숨어버림(hiding/shadowing)
    '''   
    say = '일하는 동물' 

    #Method
    def __init__(self):
        print("Employee 생성자")
        
        # 부모자식 둘다 같은 Method를 갖는 경우 (hiding/shadowing)
    def printInfo(self): 
        print(f'Employee 클래스의 printInfo 호출됨') 


    def ePrintInfo(self):
        '''
        self의 주소인 emp의 주소를 달고 돌아아님.
        1. self.subject -> Employee클래스에서 찾기 있으면 출력
        2. self.say, self.nai 
            -> Employee클래스에서 찾기 없어 
            -> (부모)Person클래스로 올라가서 찾기 있으면 출력
        3. self.helloMethod(), self.printInfo() 
            -> ''
        '''
        print(self.subject, self.say, self.nai)
        self.helloMethod()
        self.printInfo()
        '''
            super(). : 
            - 무조건 '바로 이전'의 부모클래스를(Mamber Filed,Method) 호출, 
            - 다이렉트로 바로위보다 위의 부모클래스를 부를 수 없다.
        '''
        print(super().say)
        super().printInfo()
            # private 멤버 호출 해보기
        # print(self.__msg) # Porsen 클래스의 private멤버 이기 때문에 err!
        

# 인스턴스 생성
emp = Employee()
print(emp.subject, emp.nai, emp.say)
emp.ePrintInfo()


#--------------------------------------------------------------------------
# 자식클래스 생성(Employee와는 별개의 자식 클래스.)
print('-'*100)
class Worker(Person):

    #Method
    '''
    Worker생성자 
    - 자동으로 self, nai 붙어서 생김
    - Person의 생성자가 nai를 받아야 하기 때문에, 
    - 생성자가 오버로딩이 되지 X, 오버라이딩 해야 하기 때문
    '''
    def __init__(self, nai):
        print('Worker 생성자')
        super().__init__(nai) # '부모클래스 생성자 호출'하고 싶으면 명시적으로 표시하는법.

    def wPrintInfo(self):
        print('Worker - wPrintInfo 처리')
        # self.printInfo() # Worker는  printInfo()를 갖고 있지 않기 때문에 쓰지 않는게 좋다
        super().printInfo()

wor = Worker('30')
print(wor.say, wor.nai)
wor.wPrintInfo()


#----------------------------------------------------
# 자식의 자식클래스 생성
print('='*100)
class Programmer(Worker):

    '''
    Programmer생성자 
    - 자동으로 self, nai 붙어서 생김
    - Worker 생성자가 nai를 받아야 하기 때문에,
    '''
    def __init__(self, nai):
        print('Programmer 생성자')
        super().__init__(nai)       # Bound call(super로 호출)
        Worker.__init__(self,nai)   # UnBound call(class명으로 호출)

    def pPrintInfo(self):
        print('Programmer - pPrintInfo 처리하였음')


    '''
    * method overriding
        - 부모걸 오버라이딩이라고 했다.
        - 부모 메소드와 동일 메소드 선언 하는것
        - 상속에만 나오는 단어
    '''
    def wPrintInfo(self):
        print('Programmer 에서 *method overriding함,')

# 인스턴스 생성   
pro = Programmer(35)
print(pro.say, pro.nai)
pro.pPrintInfo()
pro.wPrintInfo()

#----------------------------------------------------
# 클래스 타입 확인
print('-'*100)
print('클래스 타입 확인')
a = 3; print(type(a))       # <class 'int'>
print(type(pro))            # <class '__main__.Programmer'>
print(type(wor))

''' 
Person의 부모클래스 = <class 'object'> 
    - 우리가 만든 class는 이미 오브젝트를 상속하고 있음. 
    = object의 자식
'''
print(Person.__bases__)     # <class 'object'>

# 직접 만든 class
print(Employee.__bases__)   # <class '__main__.Person'>
print(Worker.__bases__)     # <class '__main__.Person'>
print(Programmer.__bases__) # <class '__main__.Worker'>