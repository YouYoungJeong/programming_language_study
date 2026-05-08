'''tf23
    MNIST (Modified National Institute of Standards and Technology) 데이터셋
        60,000개의 훈련 이미지와 10,000개의 손글씨 숫자 테스트 이미지를 포함합니다.
        데이터 세트는 28x28 픽셀 크기의 흑백 이미지로 구성됩니다.
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# TensorFlow oneDNN 최적화 로그/기능 끄기 - import tensorflow as tf보다 위에적용
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# MNIST 읽어오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0])
print(y_train[0])

# 풀어서 확인해보기
# for i in x_train[0]:
#     for j in i:
#         # 표준 출력장치로 출력하기
#         sys.stdout.write('%s  '%j)
#     sys.stdout.write('\n')

# 실제로 확인하기
# plt.imshow(x_train[0], cmap='gray')
# plt.show()

# 모델 만들기 선행작업 (전처리)===============================================
# 1. 구조 변경하기 (reshape) : 3차원(60000, 28, 28) ->  2차원(60000, 784)으로 만들기
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
print(x_train[0], x_train.shape) # (60000, 784)

# 2. 정규화 : 필수는 아니지만 모델 성능이 향상됨.
x_train = x_train / 255.0
x_test /= 255.0
print(x_train[0])
print(set(map(int, y_test))) 
# {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} -> softmax 사용 -> OneHot처리 해야겠네!

# 3. label OneHot처리하기 -> softmax사용하기 위해
# 정수 라벨을 One-Hot Encoding하여 밀집 배열 형태의 클래스 벡터로 변환
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(y_train[0]) # 5 => [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

# 4. validation data 직접 구성
x_val = x_train[50000:60000] # 10000 개는 validation데이터로 사용
y_val = y_train[50000:60000] 
x_train = x_train[0:50000] # 50000 개는 train 데이터로 사용
y_train = y_train[0:50000] 
print(x_val.shape, x_train.shape) # (10000, 784) (50000, 784)

# 모델 생성하기 =============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input

model = Sequential()
model.add(Input(shape=(784, )))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=32 , activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=10 , activation='softmax'))
print(model.summary())

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train, y_train, epochs=20, batch_size=128, 
    validation_data=(x_val, y_val), verbose=2
)

score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)

print(f'loss : {score[0]:.4f}, acc : {score[1]:.4f}')

# 시각화 하기
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='val accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# 모델 저장하기 - checkpoint 사용하는게 좋아
model.save("tf23model.keras")

# 모델 읽기
mymodel = tf.keras.models.load_model('tf23model.keras')

# 예측하기
pred = mymodel.predict(x_test[0:1])
print(f'pred : {pred}')
print(f'예측값 : {np.argmax(pred ,axis=1)[0]}')
print(f'real : {np.argmax(y_test[0])}')