## 동등한 폴더에서 가져올땐 같은폴더안에 있는 import도 바뀌어야함.


##경로지정 방법3-2 from , as (이름 확인 할때 ctrl + space)
print('-'*100,'\n경로지정 방법3-2 : import 하위패키지(sub).모듈명')
### 하위폴더생성: pro1/pack1/subpack/sbs에 모듈을 만듦 하위폴더생성
import pack1.subpack.sbs
pack1.subpack.sbs.sbsMansae()

import pack1.subpack.sbs as nickname
nickname.sbsMansae()

##경로지정 방법4 from , as (이름 확인 할때 ctrl + space)
print('-'*100,'\n경로지정 방법4 : 현재 package와 동등한 다른 패키지 모듈읽기')
### 폴더생성: pro1/pack1_other에 모듈을 만듦 
# import ../pack1_other.mymod1  # ../ <- import에서 vscode는 인정 X from으로 가야함
from pack1_other import mymod2
mymod2.hapFunc(4,3) # Error <- 다른 편집도구들은 출력을 해줌, vscode는 터미널에서 한번 나가야함
# 실행방법 python -m pack1.ex15module_2 <- 이렇게 호출하려면 다. 상위폴더.모듈로 불러야함.
print(mymod2.hapFunc(4,3))

## -> sub패키지를 사용하는걸 추천.

print('-'*100)
import mymod3
result = mymod3.gopFunc(4,3)
print('path가 설정된 곳의 module 읽기 - result',result)



'''
https://docs.python.org/3/ -> Library reference
'''
print('end')