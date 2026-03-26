"""
한글 형태소 분석 https://konlpy.org/ko/v0.4.3/
코퍼스(Corpus, 말뭉치)
    언어 연구, AI 학습, 자연어 처리(NLP)를 목적으로 
    실제 사용된 언어(글, 말)를 컴퓨터가 읽을 수 있는 형태로 
    대규모로 수집, 가공, 저장한 언어 자료의 집합
형태소(Morpheme)는 의미를 가지는 가장 작은 단위의 말뭉치
코퍼스를 대상으로 분석하는게 한글 형태소 분석이다.

대표적인 한글 형태소 분석 라이브러리
Okt - 학습용이라 느림.
Mecab - 실무에서 가장 많이 사용함.

"""
import warnings
warnings.filterwarnings(action='ignore')

from konlpy.tag import Okt, Kkma, Komoran

text = '나는 오늘 아침에 학교에 갔다. 가는 길에 벚꽃이 피어 너무 아름다웠다' # NLP중 corpus

# Okt 실시간은 곤란함. 데이터의 양이 많을 수록 처리속도가 떨어짐
print("-"*40,"Okt","-"*40)
okt = Okt() # okt 생성자로 객체 생성
print('형태소 : \n', okt.morphs(text))
print('\n품사 태깅(붙이기) : \n', okt.pos(text))
print('\n품사 태깅+어간(붙이기) : \n', okt.pos(text, stem=True)) # 어근 원형 출력
print('\n명사 추출하기 : \n', okt.nouns(text))
print()

# Kkma
print("-"*40,"Kkma","-"*40)
kkma = Kkma() # Kkma 생성자 객체 생성
print('형태소 : \n', kkma.morphs(text))
print('\n품사 태깅(붙이기) : \n', kkma.pos(text))
print('\n명사 추출하기 : \n', kkma.nouns(text))
print()

# Komoran
print("-"*40,"Komoran","-"*40)
komoran = Komoran() # Kkma 생성자 객체 생성
print('형태소 : \n', komoran.morphs(text))
print('\n품사 태깅(붙이기) : \n', komoran.pos(text))
print('\n명사 추출하기 : \n', komoran.nouns(text))