'''tf25
Fashion-MNIST 데이터셋은 Zalando의 상품 이미지 그레이스케일 이미지를 포함하며, 
이미지 분류 작업을 위한 잘 구조화된 데이터셋을 제공합니다

전통적인 방법으로 서포트 벡터 머신(SVM) 및 다양한 기타 머신러닝 알고리즘과 같은 
이미지 분류 작업에서 딥러닝 모델을 훈련하고 평가하는 데 널리 사용됩니다. 
이미지는 CNN만 알면되고 큰이미지들은 트렌스포머를 사용하면 된다

Fashion-MNIST는 Zalando의 상품 이미지 60,000개의 훈련 이미지와 
10,000개의 테스트 이미지를 포함합니다
데이터 세트는 크기가 28x28픽셀인 회색조 이미지로 구성됩니다.
각 픽셀에는 해당 픽셀의 밝기 또는 어둡기를 나타내는 단일 픽셀 값이 연결되어 있으며, 
숫자가 높을수록 더 어둡습니다. 이 픽셀 값은 0과 255 사이의 정수입니다.
Fashion-MNIST는 머신러닝 분야, 특히 이미지 분류 작업에서 훈련 및 테스트에 널리 사용됩니다.

레이블 : 
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot
'''

import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import sys
import os
# TensorFlow oneDNN 최적화 로그/기능 끄기 - import tensorflow as tf보다 위에적용
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input

# 데이터 읽기
fasion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = fasion_mnist
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', \
            'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(set(map(int, y_test)))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# 시각화
# plt.imshow(x_train[0], cmap='gray')
# plt.show()
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[y_train[i]])
    plt.imshow(x_train[i])
plt.show()

# 정규화 하기
print(x_train[0])
x_train = x_train / 255.0
print(x_train[0])
x_test = x_test / 255.0

# 모델 생성하기
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)), # resize를 안했음
    tf.keras.layers.Flatten(), # resize를 안한경우에 사용하는 함수 사용 - 차원축소
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

print(model.summary())

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy', # 내부적으로 onehot처리해줌
    metrics=['accuracy']
)

model.fit(
    x_train, y_train, batch_size=128, epochs=10, verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'test_loss : {test_loss:.4f}, test_acc : {test_acc:.4f}')

pred = model.predict(x_test)
print(f'pred : {pred[0]}')
print(f'예측값 : {np.argmax(pred[0])}, {class_names[np.argmax(pred[0])]}')
print(f'real : {y_test[0]}, {class_names[y_test[0]]}')


# 각 이미지 출력용 함수 선언(예측 이미지와 실제 레이블 비교)
def plot_image(i, pred, y_true, x_img):
    # 예측값[i]
    pred_arr = pred[i]
    # 실제값 받아오기
    true_label = y_true[i]
    img = x_img[i]

    pred_label = np.argmax(pred_arr)
    pred_percent = 100 * np.max(pred_arr)
    color='green' if pred_label == true_label else 'magenta'
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    plt.xlabel(
        f'예측:{class_names[pred_label]} {pred_percent:.0f}%\n'
        f'실제:{class_names[true_label]}', color=color
    )

# 각 이미지에 라벨 등의 정보 표시 - 막대 그래프
def plot_values_arr(i, pred, y_true):
    # 예측값[i]
    pred_arr = pred[i]
    # 실제값 받아오기
    true_label = y_true[i]
    pred_label = np.argmax(pred_arr)
    
    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.yticks([])
    plt.ylim([0, 1])
    bars = plt.bar(range(10), pred_arr)
    bars[pred_label].set_color('magenta') # 예측값
    bars[true_label].set_color('green') # 실제값


def show_one_prediction(i, pred, y_true, x_img):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plot_image(i, pred, y_true, x_img) # 그래프 그리는 함수로 넘어가
    
    plt.subplot(1, 2, 2)
    plot_values_arr(i, pred, y_true)

    plt.tight_layout()
    plt.show()

show_one_prediction(1, pred, y_test, x_test) # 1개 이미지

# 여러 이미지 보기 3 * 3 출력
def show_prediction_grid(start, pred, y_true, x_img, rows=3, cols=3):
    plt.figure(figsize=(9, 9))

    for n in range(rows * cols):
        i = start + n
        plt.subplot(rows, cols, n+1)
        pred_label = np.argmax(pred[i])
        true_label = y_true[i]
        pred_percent = 100 * np.max(pred[i])

        color='green' if pred_label == true_label else 'magenta'
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_img[i], cmap='gray')
        plt.xlabel(
        f'예측:{class_names[pred_label]} {pred_percent:.0f}%\n'
        f'실제:{class_names[true_label]}', color=color
        )
    plt.tight_layout()
    plt.show()

# 0번 부터 9개 보기
show_prediction_grid(0, pred, y_test, x_test)

# 15번 부터 9개 보기
show_prediction_grid(15, pred, y_test, x_test)