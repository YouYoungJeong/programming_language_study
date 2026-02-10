# 우편정보 파일 자료 읽기
# 키보드에서 입력한 동이름으로 해당 주소 정보 출력
    

def zipProcess():
    donIrum = input('동이름 입력 : ')
    # donname = '도곡2동'
    # print(donIrum)

    with open(r"zipcode.txt", mode='r', encoding='euc-kr') as f:
        line = f.readline()     # 한행 읽기 -> 135-806 서울    강남구  개포1동 경남아파트              
        # print(line)
        # # lines = line.split('\t') # tap = \t
        # lines = line.split(chr(9)) 
        # print(lines)

        while line:
            lines = line.split(chr(9))
            if lines[3].startswith(donIrum):
                # print(lines)
                print(f'우편번호 : {lines[0]}, {lines[1]}, {lines[2]}')
            line = f.readline()
        
        '''
        [ascii 코드]
        자판위의 모든값이 다 있어.
        a:chr(97)~z:chr(122), A(65)~Z(90)
        enter = line feed, charriage return -> chr(10), cha(13)
        tap = chr(9)
        ascii 코드에 추가적으로 추가한 데이터가 utf-8, utf-16, euc-kr...
        '''





# 메인모듈이 맞니~?
if __name__ == '__main__':
    zipProcess()