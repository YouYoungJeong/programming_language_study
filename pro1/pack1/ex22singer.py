# Singer라는 Type만 들어있는 그릇.
class Singer:  

    title_song = '빛나라 대한민국'
    # 초기화 할게 없어서 생성자 안만듦.

    def sing(self):
        msg = "노래는 :"
        print(msg, self.title_song)