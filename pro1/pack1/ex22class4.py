# class는 새로운 타입을 만들어 자원을 공유 가능
# 속성, 행위 둘다 필요한 경우 Class를 사용
# 행위만 있는 경우 Function을 사용
''' 모듈 로 보냄
class Singer:  
    title_song = '빛나라 대한민국'
    # 초기화 할게 없어서 생성자 안만듦.
    def sing(self):
        msg = "노래는 :"
        print(msg, self.title_song)
'''
# import ex22singer 로 부르면  ex22singer.Singer()로 불러와야함
# import ex22singer as Singer # -> Singer.Singer()로 불러야함.

from ex22singer import Singer

bts = Singer() # 생성자 호출해 객체 생성 후 주소를 bts라는 변수에 치환
bts.sing()
print(type(bts))
bts.title_song = 'DNA'
bts.sing()
bts.co = '빅히트 엔터테이먼트'
print(f'소속사 : {bts.co}')

ive = Singer()
ive.sing()
print(type(ive))

# bts만 가지고 있는 객체 ive에 선언 안하면 에러
# print(f'소속사 : {ive.co}')

# prototype의 객체 수정
Singer.title_song = '아 대한민국'
bts.sing()
ive.sing()

niceGroup = ive # 주소 치환 둘이 같은거야 닉네임을 받
niceGroup.sing()