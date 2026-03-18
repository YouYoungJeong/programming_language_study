"""
배열의 행, 열 추가 .....
c_, r_, append, insert, delete, where, concatenate, hsplit, vsplit, random
"""
import numpy as np

aa = np.eye(3) # 주대각이 1인 단위행렬(스퀘어 메트릭스)
print(aa)


bb = np.c_[aa, aa[2]]  # 2열과 동일한 열 추가
print(bb)

cc = np.r_[aa, [aa[2]]] # 2행과 동일한 행 추가
print(cc)

# 비슷한 결이 reshape
print("-"*50)

"""append, insert, delete"""
a = np.array([1,2,3])
print(a)
# 추가
b = np.append(a,[4,5])
b = np.append(a,[4,5], axis=0) # axis=0 행기준
print(b)
# 삽입
c = np.insert(a, 0, [6,7])
print(c)
# 삭제
d = np.delete(a, 1)
print(d)

print()
aa = np.arange(1,10).reshape(3,3)
print(aa)
print(np.insert(aa, 1, 99))         
print(np.insert(aa, 1, 99, axis=0)) # 행기준
print(np.insert(aa, 1, 99, axis=1)) # 열기준
print("-"*50)

'''조건 연산 where(조건, 참, 거짓)'''
x = np.array([1,2,3])
y = np.array([4,5,6])
conditionData = np.array([True, False, True]) # 조건(bool)을 가지고 있는 변수
# 참이면 x값을 , 거짓이면 y값을 출력하라
result = np.where(conditionData, x, y)
print(result)
print()

aa = np.where(x >= 2)
print(aa) # (array([1, 2]),)인덱스 출력
print(a[aa])
print("-"*50)

'''배열 결합/분할'''
# 결합
kbs = np.concatenate([x, y])
print(kbs)
print()
# 분할
mbc, sbs = np.split(kbs, 2)
print(mbc ,' ', sbs)

# 배열 좌우로 분할
print()
a = np.arange(1,17).reshape(4,4)
print(a)
print()
x1, x2 = np.hsplit(a, 2)
print(x1,"\n",x2)
print()
# 배열 상하로 분할
x1, x2 = np.vsplit(a, 2)
print(x1, "\n",x2)

"""표본 추출(sampling) - 샘플링 해야하는 상황이 굉장히 많아.
복원 - 꺼내고 넣고 가능 (중복 O)
비복원 - 꺼냈으면 끝이다(중복 X)
"""
li = np.array([1,2,3,4,5,6,7])

# 복원
for _ in range(5):
    print(li[np.random.randint(0, len(li)-1)], end=' ')
print()

# 비복원
import random
print(random.sample(li.tolist(), 5)) # random.sample()함수는 리스트타입으로 맞춰줘야함.

print()

# choice
print(np.random.choice(range(1, 46), 6))
print(np.random.choice(range(1, 46), 6, replace=True))  # 복원
print(np.random.choice(range(1, 46), 6, replace=False)) #비복원