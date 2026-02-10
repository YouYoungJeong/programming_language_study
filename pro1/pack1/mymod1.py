# pack1/mymod.py
## 전역변수와 함수만 생성 -> 자기혼자선 쓸 일이 없고 호출해야 사용.

tot = 100  # 전역 변수

def listHap(*ar): # *튜플
    print(ar)
    if __name__ == '__main__':
        print('나는 메인모듈')


def kbs():
    print('대한민국 대표방송')

def mbc():
    print('문화방송')