# with 구문 사용 - 내부적으로 close() 함

try:
# 파일 저장
    with open('ftext3.txt', mode='w', encoding='utf-8') as fobj1:
        fobj1.write('파이썬 에서 문서저장\n')
        fobj1.write('with 구문은\n')
        fobj1.write('명시적으로 close() 할 필요 X')
    print('저장 완료')

# 파일 읽기
    with open('ftext3.txt', mode='r', encoding='utf-8') as fobj2:
        print(fobj2.read())
    
except Exception as e:
    print('err : ', e)

# 피클링
print("\n\n")
print('피클링(일반 객체 및 복합 객체 파일 처리- 부분적으로 저장하고 싶을때)')
import pickle

try:
    dictData =  {'tom':"111-1111", '길동':'222-2222'}
    listData = ['마우스', '키보드' ]
    tupleData = (dictData, listData)

    with open("hello.dat", mode='wb') as fobj3: # 객체로 저장 할때는 dat로 주자.(또는 obj)
        pickle.dump(tupleData, fobj3)           # pickle.dump(대상, 파일객체)
        pickle.dump(listData, fobj3)            # list type 객체만 저장
    print('특정 객체를 파일을 객체로 저장')

    print('피클 객체 읽기')
    with open("hello.dat", mode='rb') as fobj4:
        # 첫번째 정리된게 먼저 나옴, 읽을때 load(파일객체명)
        a, b = pickle.load(fobj4) 
        print(f'a : {a}')
        print(f'b : {b}')
        c = pickle.load(fobj4)
        print(f'c : {c}')

except Exception as e:    
    print('err : ', e)