'''
다항분류(Multi-class Classification)는 마지막 출력층 활성화 함수에 softmax를 사용함.
다항분류도 이진분류도 사용가능.

소프트맥스(Softmax) 함수
    입력받은 실수 벡터를 0~1 사이의 확률값으로 정규화하여, 
    모든 출력의 합이 1이 되도록 만드는 함수
    지수함수 / 지수함수 합 => 다합치면 1
    
    기본적으로 ** label을 무조건 OneHot처리를 해야한다. **
    ->안하는 경우도 있음
'''

# softmax함수
import numpy as np
def softmaxFunc(a):
    c = np.max(a) # 너무 큰값에 err가 나서 변수 선언 해줘야함.
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

data = np.array([0.3, 2.8, 4.0])
print(softmaxFunc(data)) # [0.01864635 0.22715905 0.7541946 ]

# 다항분류 모델 생성
#   출력은 softmax 활성함수로 인해 복수개의 확률값으로 출력. 
#   이 때 가장큰 인덱스를 분류결과로 취함
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical # One-Hot 지원
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
np.random.seed(1)
xdata = np.random.random((1000, 12))  # 1000행 12열 난수 발생 : feature(시험점수라고 가정)
ydata = np.random.randint(5, size=(1000, 1)) # category 5가지 : label(과목이라고 가정)
print(xdata[:2])
print(ydata[:2]) # [[2] [0]]

# label OneHot처리하기
ydata = to_categorical(ydata, num_classes=5)
print(ydata[:2]) # [[0. 0. 1. 0. 0.] [1. 0. 0. 0. 0.]]

# model 생성하기
model = Sequential()
model.add(Input(shape=(12, ))) # feature수
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
print(model.summary()) # Total params: 1,029

# compile
model.compile(
    optimizer='adam',

    # Categorical Crossentropy는 다중 클래스 분류(Multi-class Classification) 문제에서 
    # 모델의 출력값과 실제 정답 사이의 오차를 계산하는 데 사용하는 손실 함수(Loss Function)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# fit
history = model.fit(xdata, ydata, epochs=2000, batch_size=32, verbose=2, shuffle=True)

# 평가
model_eval = model.evaluate(xdata, ydata, verbose=0)
print('모델 평가 결과 :', model_eval)

# 시각화 하기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'])
ax1.set_title('Loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')

ax2.plot(history.history['accuracy'])
ax2.set_title('Accurcy')
ax2.set_xlabel('epoch')
ax2.set_ylabel('acc')
plt.show()

# 기존 값으로 분류 예측하기
print('예측값 : ', model.predict(xdata[:5]))
# [[0.08733685 0.19838686 0.46642578 0.24596517 0.00188541]
#  [0.74652314 0.06846038 0.12338578 0.06163027 0.00000041]
#  [0.0067992  0.1740631  0.00020479 0.02008468 0.7988483 ]
#  [0.14257699 0.00051384 0.00043869 0.00056481 0.85590565]
#  [0.00897905 0.894763   0.00001077 0.09623523 0.00001203]]

print('실제값 : ', ydata[:5])
#  [[0. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0.]]

# 인덱스로 출력하기
print('예측값(인덱스로 출력) : ', np.argmax(model.predict(xdata[:5]), axis=1))
print('실제값(인덱스로 출력) : ', [int(i) for i in np.argmax(ydata[:5], axis=1)])
print()

# 새로운 값으로 예측하기
x_new = np.random.random([1, 12])
print(x_new)
new_pred = model.predict(x_new)
print('분류 결과 확률값 :', new_pred)
print('분류 결과 합 :', np.sum(new_pred))
print('분류 결과 :', np.argmax(new_pred))

# 예측 결과 과목명으로 출력
classes = np.array(['국어','영어','수학','과학','체육'])
print('예측값(과목명으로 출력) : ', classes[np.argmax(new_pred)])
print('예측값(과목명으로 출력) : ', classes[np.argmax(new_pred, axis=1)[0]])