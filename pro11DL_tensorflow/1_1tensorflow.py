'''
Tensorflow
'''
import numpy as np
import tensorflow as tf # 한글 깨질때 터미널 : chcp 65001 실행

print(tf.__version__)
print('즉시 실행 모드 :', tf.executing_eagerly())
print("GPU 사용 정보 확인 :", tf.config.list_physical_devices("GPU")) # []
print()

print("Tensor : 텐서플로에서 데이터를 담는 기본 자료구조 (숫자 데이터 저장용 다차원 배열)")
# ndarray와 유사하지만 tensorflow에서 연산에 사용되도록 만들어진 객체.
print()

# ============================================
# Tensor의 차원
# ============================================

print(12, type(12))     # python 상수로 python이 직접 계산 - 12 <class 'int'>

# 0d(0차원) 텐서 (scaler) - tf.Tensor(1, shape=(), dtype=int32)
print(tf.constant(1))   

# 1d(1차원) 텐서 (vector) - tf.Tensor([12], shape=(1,), dtype=int32)
print(tf.constant([12]))   

# 2d(2차원) 텐서 (matrix) - tf.Tensor([[12]], shape=(1, 1), dtype=int32)
print(tf.constant([[12]]))

# 2d(2차원) 텐서 (matrix) - tf.Tensor([[12  1]], shape=(1, 2), dtype=int32)
print(tf.constant([[12, 1]]))   

# 차원 확인(rank)
print(tf.rank(tf.constant([[12, 1]]))) # tf.Tensor(2, shape=(), dtype=int32)

# ============================================
# Tensorflow의 print
# ============================================
print('print : python의 기본 함수인 print - 객체 자체를 문자열로 변환 후 출력, 정보 중심 출력')
tf.print("tf.print : Tensorflow전용 출력함수 tf.print - Tensor 실제값을 중심으로 출력")
tf.print(12, type(12)) # 12 <class 'int'>
print()

# ============================================
# Tensor연산
# ============================================

# np.array : 일반 수치 연산(CPU연산이 기본, 자동 미분 불가, 값변경 가능)
imsi = np.array([1, 2]) 
print(type(imsi)) # <class 'numpy.ndarray'>
imsi[0] =  10 # 값 변경 가능 

# Tensor : 딥러닝 연산(GPU연산도 가능, 자동 미분 가능, 값변경 불가능(constant))
a = tf.constant([1, 2]) 
print(type(a)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
# a[0] = 10
# TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
b = tf.constant([3, 4])

# Tensor요소값 더하기 (열단위 연산), 사칙연산 가능
c = a + b
tf.print(c) # [4 6]

# Broadcast 연산 - numpy 기반이라서!
d = tf.constant([3])
e = c + d
tf.print(e) # [7 9]
print()

# ============================================
# numpy와 Tensorflow 형변환
# ============================================

print("numpy와 Tensorflow 형변환")
print(7)                        # 7
print(tf.convert_to_tensor(7))  # tf.Tensor(7, shape=(), dtype=int32)
print(tf.constant(7).numpy())   # 7
# tf.add(), tf.sub
arr = np.array([1, 2])
tfarr = tf.add(arr, 5)
print(tfarr, type(tfarr))   # Tensor연산을 하면 Tensor타입으로 자동 형변환됨
# tf.Tensor([6 7], shape=(2,), dtype=int64) 
# <class 'tensorflow.python.framework.ops.EagerTensor'>
print(np.add(tfarr, 2))     # [8 9] 배열 연산을 하면 numpy타입으로 자동 형변환됨
print()

# ============================================
# Tensorflow 변수 선언후 사용하기
# tf.Variable() : Tensorflow에서 값이 바뀔 수 있는 Tensor를 만들때 사용.
# assign() : 변수값 수정하기
#   예) weight, bias ...
# ============================================
print('Tensorflow 변수 선언후 사용하기')
# 변수에 값 담기(기억)
v1 = tf.Variable(1.0)
tf.print('v1 :', v1) # 1

v2 = tf.Variable(tf.ones((2, ))) # 1으로 채워진 변수
tf.print('v2 :', v2) # [1 1]

v3 = tf.Variable(tf.zeros((2, ))) # 0으로 채워진 변수
tf.print('v3 :', v3) # [0 0]
print()

# 변수값 수정하기(assign()) = Tensor연산
# v1 = 123 # Error
# tf.print('v1 :', v1) # AttributeError: 'int' object has no attribute 'assign'

v1.assign(123) # 변수값 변경
tf.print('v1 :', v1) # 123

v2.assign([30, 40])
tf.print('v2 :', v2) # [30 40]
print()


aa = tf.Variable(tf.zeros((2, 1))) # 2행 열에 모두 0을 기억
tf.print('aa :', aa) # [[0] [0]]
aa.assign(tf.ones((2, 1)))
tf.print("aa :", aa) # [[1] [1]]

# ============================================
# Tensorflow 변수 사칙연산
# ============================================

# 더하기 치환
aa.assign_add([[2],[3]])
tf.print("aa :", aa) # [[3] [4]]
# 빼기 치환
aa.assign_sub([[2],[3]])
tf.print("aa :", aa) # [[1] [1]]
# 곱하기 치환
aa.assign(aa * [[2],[3]]) #  assign_mul X
tf.print("aa :", aa) # [[2] [3]]
# 나누기 치환
aa.assign(aa / [[2],[3]]) # #  assign_div X
tf.print("aa :", aa) #  [[1] [1]]

# ============================================
# Tensorflow 난수 처리
# ============================================
print(tf.random.uniform([1], 0, 1)) # 균등분포 :([shape],min,max)
print(tf.random.normal([3], 0, 1)) # 정규분포 :([shape],평균,표준편차)
print(tf.random.normal([3, 2], mean=0, stddev=1)) # 정규분포 :([shape],평균,표준편차)