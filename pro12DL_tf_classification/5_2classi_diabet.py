'''
이항분류(sigmoid)는 다항분류(softmax)로 처리 가능
'''
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

datas = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv',
                    delimiter=',')
print(datas.shape) # (768, 9)
print(datas[:1]) # [[  6.   148.    72.    35.     0.    33.6    0.627  50.    1.  ]]
print(set(datas[:,-1])) # {np.float64(0.0), np.float64(1.0)}

x_train, x_test, y_train, y_test = train_test_split(
    datas[:, 0:8], datas[:, -1], test_size=0.3, shuffle=True, random_state=123)
print(x_train.shape, x_test.shape) # (537, 8) (231, 8)
print()

# 이항분류(sigmoid)
model = Sequential()
model.add(Input(shape=(8, )))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)
model.fit(
    x_train, y_train, epochs=100, batch_size=30, verbose=0
    )
scores = model.evaluate(x_test, y_test, verbose=0)
print(f'sigmoid scores : {scores}')
print()

# 다항분류(softmax)
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[:3])
model = Sequential()
model.add(Input(shape=(8, )))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)
model.fit(
    x_train, y_train, epochs=100, batch_size=30, verbose=0
    )
scores = model.evaluate(x_test, y_test, verbose=0)
print(f'softmax scores : {scores}')
print()