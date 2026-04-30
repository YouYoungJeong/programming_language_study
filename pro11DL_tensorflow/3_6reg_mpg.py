'''
다중선형회귀
    조기종료(early stopping) 무조건 넣어야 하는 callback1(두번째는 checkpoint)
    자동차 연비 예측 - 선형.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras import optimizers

datas = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/auto-mpg.csv", 
                    na_values='?') # ?가 있는 데이터는 결측치 처리를 할거야.
print(datas.head(2))
print(datas.info())

del datas['car name']
datas = datas.dropna()
print(datas.isna().sum())
print(datas.head(2))

datas.drop(['cylinders','acceleration','model year','origin'], axis='columns', inplace=True)
print(datas.head(2))

# 데이터 분포 확인하기
# sns.pairplot(datas[['mpg','displacement','horsepower','weight']],
#                 diag_kind='kde') # 밀도기반
# plt.show()

# train / test split 직접 나누기
# feature / label 나누지 말아야함
# train dataset sampling하기
train_dataset = datas.sample(frac=0.7, random_state=123)
print(train_dataset[:2], train_dataset.shape) # (274, 4)
# train dataset만큼 없애면 test dataset - test는 sampling하면 안됨- 중복 가능성O
test_dataset = datas.drop(train_dataset.index)
print(test_dataset[:2], test_dataset.shape) # (118, 4)

# 표준화(StandardScale) : (요소값 - 평균) / 표준편차 직접하기
train_stat = train_dataset.describe()
train_stat.pop('mpg')   # mpg분리하기
print(train_stat)
train_stat = train_stat.T # 전치
# train_stat = train_stat.transpose() # == 전치
print(train_stat)

# scale 함수 생성하기
def stdscale_func(x):
    return (x - train_stat['mean']) / train_stat['std']

# print(stdscale_func(train_dataset[:3]))

# 표준화한 feature 추출
# train_x
st_train_data = stdscale_func(train_dataset)
st_train_data = st_train_data.drop(['mpg'], axis='columns')
print(st_train_data[:3])
# test_x
st_test_data = stdscale_func(test_dataset)
st_test_data = st_test_data.drop(['mpg'], axis='columns')
print(st_test_data[:3])

# label추출
train_label = train_dataset.pop('mpg')
print(train_label[:3])
test_label = test_dataset.pop('mpg')
print(test_label[:3])

# model 생성 - function으로 작업
def build_model():
    network = Sequential([ # 시퀀셜은 순차적으로만 가능
        Input(shape=(3, )),
        Dense(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='linear'),
        # Dense(units=1), <- linear
    ])
    opti = tf.keras.optimizers.Adam(learning_rate=0.001)
    network.compile(optimizer=opti, loss='mean_squared_error', 
                    metrics=['mean_squared_error','mean_absolute_error'])
    return network
model = build_model()
print(model.summary())

# 조기 종료 EarlyStopping() 객체 생성
EPOCHS = 5000
early_stop = tf.keras.callbacks.EarlyStopping(
        # monitor = 뭘 기준으로 정할지를 결정.(loss, val_loss)
        # fit에 validation을 안주면 loss밖에 못씀
        # fit에 validation을 주면 val_loss도 사용가능하면 val_loss를 사용하길 권장함.
    monitor='val_loss',     # 회귀 , 분류는 - acc
        # patience = 몇번의 epoch까지 참을 수 있어(기다릴지.) 
        # 실무에서 보통 10정도 줌
    patience= 5,
        # baseline = 최소한의 성능
    # baseline=0.0001 # 작게 주는게 좋다.
        # restore_best_weights = 학습중 가장 성능이 조은 epoch의 가중치를 취함
        # 최적의 w / b 값을 찾아야함
    restore_best_weights=True
)

history = model.fit(st_train_data, train_label, batch_size=32,
                    epochs=EPOCHS, verbose=2,
                    validation_split=0.2, # st_train을  8:2로 나눔
                    callbacks=[early_stop] 
)
df = pd.DataFrame(history.history) 
print(df.head(2))
print(df.columns)
# ['loss', 'mean_absolute_error', 'mean_squared_error', 'val_loss',
#        'val_mean_absolute_error', 'val_mean_squared_error']

# 모델 학습정보 시각화 하기
def plt_history(df):
    hist = df
    hist['epoch'] = history.epoch
    # print(hist.head(5))
    
    #시각화하기
    plt.figure(figsize=(8, 12))
    # MAE
    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.ylabel('MAE [mpg]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='train err')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='vaildation err')
    plt.legend()
    # MSE
    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.ylabel('MSE [mpg]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='train err')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='vaildation err')
    plt.legend()
    plt.show()

plt_history(df)

# 모델 평가하기(R2)
loss, mse, mae = model.evaluate(st_test_data, test_label)
print(f'loss : {loss:.3f}')
print(f'mse : {mse:.3f}')
print(f'mae : {mae:.3f}')
# loss : 15.041
# mse : 15.041
# mae : 3.172
print('R2(결정계수, 설명력) :', r2_score(test_label, model.predict(st_test_data)))
#  -0.2743784 : 결정계수가 음수가 나온 이유
# 어딘가에서 설계가 잘못되었다는 뜻
# 조기종료 baseline 삭제 :  0.735552166 : 너무 빨리 끝나서
# learning_rate = 0.001로 조절 : 0.7235237
# restore_best_weights=True 조기종료 옵션 추가 : 0.726395

# 새로운 값으로 예측
new_data = pd.DataFrame({
    'displacement' : [300, 400],
    'horsepower' : [120, 150],
    'weight' : [2000, 4000]
})
new_st_data = stdscale_func(new_data)
new_data_pred = model.predict(new_st_data).ravel()
print(f'새 mpg 예측결과 : {new_data_pred}') #  [20.811623 16.989494]