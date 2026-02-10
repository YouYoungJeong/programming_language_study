###모듈 생성시 편집기에 따라 약간 달라짐(VSCode, Pycham...)
# pro1/pack1/ex15module.py : main
# 사용자 정의 모듈 처리하기

# pro1/pack1/mymod1.py 을 생성하고옴
print('사용자 정의 모듈 처리하기')

s = 20 # 뭔가 하다가 모듈 필요
##경로지정 방법1
print('\n경로지정 방법1 : improt 모듈명')
import mymod1
print(dir(mymod1)) # dir 모듈 멤버확인하는 명령어
'''
['__builtins__', '__cached__', '__doc__', '__file__','__loader__', '__name__', '__package__', '__spec__', 
: 기본적으로 python이 만들어줌
'kbs', 'listHap', 'mbc', 'tot' : 내가 만든 맴버]
''' 

print(mymod1.__file__) # 파일의 경로
print(mymod1.__name__) # 모듈명
### 튜플로 만들라는 모듈에 넣어서 리스트 출력해보기.
list1 = [1, 2]
list2 = [3, 4, 5]
mymod1.listHap(list1, list2) # 불려서 메인이 X, 

if __name__ == '__main__': print('와우 메인모듈')


##경로지정 방법2 from , as (이름 확인 할때 ctrl + space)
print('-'*100,'\n경로지정 방법2 : from 모듈명 improt 함수명 또는 전역변수명')
from mymod1 import kbs
kbs()

from mymod1 import mbc, tot
mbc()
print(tot)

from mymod1 import * # *을 사용해 mymod1모듈의 모든 멤버로딩(비권장)
print('tot:', tot)

from mymod1 import mbc as 엠비씨만세별명 # 별명 주기
엠비씨만세별명()

##경로지정 방법3-1 from , as (이름 확인 할때 ctrl + space)
print('-'*100,'\n경로지정 방법3-1 : import 하위패키지(sub).모듈명')
### 하위폴더생성: pro1/pack1/subpack/sbs에 모듈을 만듦 하위폴더생성
import subpack.sbs
subpack.sbs.sbsMansae()

import subpack.sbs as nickname
nickname.sbsMansae()

## 경로지정 방법 5. 직접 Lib파일에 넣기
import mymod3
result = mymod3.gopFunc(4,3)
print('path가 설정된 곳의 module 읽기 - result',result)
print('end')


