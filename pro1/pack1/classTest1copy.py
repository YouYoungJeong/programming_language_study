'''
Machine 조건 : 
입력자료 키보드 사용(input)
커피 1잔 200원
100원 넣고 커피 요구하면 요금 부족 메세지 출력
400원 넣고 2잔 요구하면 두잔 출력
500원 넣고 1잔 요구하면 300원 반납


CoinIn :
출력형태-------
동전을 입력하세요 : 400
몇 잔을 원하세요 : 2
커피 2잔과 잔돈 100원

'''


class Machine:
    cupCount = 1
    def __init__(self):
        self.machin = CoinIn()

    def showData(self):
        coin = int(input('동전을 입력하세요'))
        cupCount = int(input('몇잔을 원하세요'))
        coin = self.machin.culc(coin, cupCount)
    

class CoinIn:
    coin =0
    change =  0
    
    def culc(self, coin, cupCount):
        totcoffee = cupCount*200
        if coin < 200:
            print('금액이 부족합니다.')            
        elif coin-totcoffee < 0:
            print('금액이 부족합니다.')
        else:
            return print(f'커피 {cupCount}잔과 잔돈{coin -(totcoffee)}원') 


Machine().showData()