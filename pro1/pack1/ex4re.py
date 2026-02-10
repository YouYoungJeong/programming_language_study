# 정규표현식 
import re # 정규표현식 지원 모듈 로딩
ss= "1234 abc가나다abcABC_1123555집에가나78요_6'Python is fun'"
print(ss)
# Python의 정규표현식은  r을 선행하라
print(re.findall(r'123', ss)) # return Type는 list
print(re.findall(r'가나', ss))
print(re.findall(r'[0-9]', ss)) # 숫자만 뽑고 싶으면 옵션
print(re.findall(r'[0-9]+', ss)) # 한개 이상 붙어져 있으며 붙어져서 나옴
print(re.findall(r'[0-9]*', ss)) # 0개 이상 붙어져 있으며 붙어져서 나옴
print(re.findall(r'[0-9]{2,3}', ss)) # 2~3개 이상 붙어져 있을때 > 전화번호 찾을때 사용. 정규식 사용이 편해
print(re.findall(r'[a-zA-Z]+', ss))
print(re.findall(r'[가-힣]+', ss))
print(re.findall(r'\d+', ss)) # 숫자
print(re.findall(r'\D+', ss)) # 문자