'''tf20
Label을 OntHot처리의 유무에 따른 compile 방법
    loss='sparse_categorical_crossentropy' -> label을 OneHot을 내부적으로 처리
    loss='categorical_corssentropy'        -> label을 OneHot이 선행되어 있어야함.
    직접 OneHot처리를 하는걸 권장함.
+ 혼동행렬

Zoo Animal dataset
animal_name: Unique for each instance - 각 인스턴스마다 고유합니다.
hair Boolean                            헤어 불리언
feathers Boolean                        깃털 불리언
eggs Boolean                            계란 불리언
milk Boolean                            우유 불리언
airborne Boolean                        공중 부울
aquatic Boolean                         수생 부울
predator Boolean                        포식자 불리언
toothed Boolean                         톱니형 부울
backbone Boolean                        백본 불리언
breathes Boolean                        숨쉬다 불리언
venomous Boolean                        독성 부울
fins Boolean                            지느러미 부울
legs Numeric (set of values: {0,2,4,5,6,8})다리 숫자 (값 집합: {0,2,4,5,6,8})
tail Boolean                            꼬리 부울
domestic Boolean                        국내 부울
catsize Boolean                         고양이 크기 불리언
class_type Numeric (integer values in range [1,7]) 클래스 유형은 숫자형(1, 7 범위의 정수 값)입니다.
1:포유류, 2번:조류 ...
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


datas = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/zoo.csv")
print(datas.head(3))
print(datas.info())

x_data = datas.iloc[:, :-1].astype('float32').values
y_data = datas.iloc[:, -1].astype('int32').values
print(x_data[0], x_data.shape) # (101, 16)
print(y_data[0],sorted(set(map(int, y_data))) ,y_data.shape) # [0, 1, 2, 3, 4, 5, 6] (101,)

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
print(x_train.shape, y_train.shape) # (80, 16) (80,)
nb_classes = len(set(y_data))

model = Sequential([
    Input(shape=(x_data.shape[1], )),
    Dense(units=64 ,  activation='relu'),
    Dropout(rate=0.3),
    Dense(units=32 ,  activation='relu'),
    Dropout(rate=0.3),
    Dense(units=nb_classes,  activation='softmax')
])
print(model.summary())

# label OneHot처리 안한경우
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # label을 OneHot을 내부적으로 처리
    # loss='categorical_corssentropy', -> label을 OneHot이 선행되어 있어야함.
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'loss : {loss:.4f}, acc : {acc:.4f}')

# 시각화(loss, acc)
plt.figure(figsize=(12, 6))
# loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='loss', color='pink')
plt.plot(history.history['val_loss'], label='val_loss', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.legend()
# acc
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy', color='pink')
plt.plot(history.history['val_accuracy'], label='val_accuracy', color='darkgreen', alpha=0.9)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

# 혼동행렬 출력
y_pred = np.argmax(model.predict(x_test), axis=1)
print(f"classification_report : \n{classification_report(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

import seaborn as sns
# 혼돈행렬 heatmap출력
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('predicted')
plt.ylabel('True')
plt.show()

# 새로운 값으로 분류 예측
new_data = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 4., 0., 0., 1.]], dtype='float32')
probs = model.predict(new_data)
pred_class = np.argmax(probs)
print('분류 예측 확률 :', probs.ravel())
print('분류 예측 라벨 :', pred_class)