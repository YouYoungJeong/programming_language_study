# 추상
# 이름은 Employee만 출력
# 이름나이받아서 월급계산
#  Temporary로받아서 Employee를 data_print() 찍어서 출력 -> t.data_print(), r.data_prin(), s.data_print()
# Temporary로 일수일당 가지고 있어
# 일당 계산은 Employee의 pay가 함 오버라이드 강요 계산은 오버라이딩해서 계산

from abc import *

class Employee:
    def __init__(self, irum, nai):
        self.irum = irum
        self.nai = nai
    
    @abstractmethod
    def pay(self):
        pass
        
    @abstractmethod
    def data_print(self):
        print(Employee().irunnai_print + self.pay())

    def irunnai_print(self): # 이름 나이 출력용
        print('이름 : '+ self.irum+' , ' + '나이 :' + str(self.nai), end=' ')

class Temporary(Employee):
    def __init__(self, irum, nai, ilsu, ildang):
        Employee.__init__(self, irum, nai)
        self.lisu = ilsu
        self.ildang = ildang

    def pay(self):
        return self.ildang*self.lisu         
    
    def data_print(self):
        self.irunnai_print()
        print(f'월급 : {self.pay()}')
        # print(f'{Employee.irunnai_print(self)} 일수 :, 월급 :')




class Regular(Employee):
    salary = 0
    def __init__(self, irum, nai, serlary):
        Employee.__init__(self, irum, nai)
        self.serlary = serlary

    def pay(self):
        pass
    
    def data_print(self):
        self.irunnai_print()
        print(f'급여 : {self.serlary}')

t = Temporary('홍길동', 25, 20, 150000)
t.data_print()
r = Regular('한국인', 27, 3500000)
r.data_print()

class Salesman(Regular):
    sales = ""
    commission = 0

    def __init__(self, irum, nai, serlary, sales, commission):
        Regular.__init__(self, irum, nai, serlary)
        self.sales = sales
        self.commission = commission
        

    def pay(self):
        return self.serlary +(self.sales * self.commission)
    
    def data_print(self):
        super().irunnai_print()
        print(f'수령액 : {int(self.pay())}')

s = Salesman('손오공', 29, 1200000, 5000000, 0.25)
s.data_print()