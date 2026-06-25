# RNN으로 글자 단위 학습 후 영문 생성
import os, sys, random, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

file_name = 'rnn6text.txt'
with open(file_name, encoding='utf-8') as f:
    et = f.read().lower()

print(et[:300] if len(et) > 300 else et)

# 문자 단위 어휘집 생성
chars = sorted(list(set(et))) # 고유 문자 정의
print(chars)

# 문자에 대한 정수 mapping
char_to_int = {c:i for i, c in enumerate(chars)}
print(char_to_int)

int_to_char =  {i:c for i, c in enumerate(chars)}
print(int_to_char)

n_chars = len(et) # 전체 문자열 수
n_vocab = len(chars)
print(f"전체 글자 수 : {n_chars}")
print(f"전체 어휘 크기 : {n_vocab}")
print()

# 시퀀스 구성
seq_length = 10 # 입력 window 길이 (이전 10글자로 다음 글자 한개를 예측)
dataX, dataY = [], []

# 학습용 시퀀스 제작
for i in range(0, n_chars - seq_length, 1):
    seq_in = et[i:i + seq_length]   # 입력 문자열
    seq_out = et[i + seq_length]    # 다음 글자 예측
    dataX.append([char_to_int[ch] for ch in seq_in]) # 입력을 숫자 시퀀스로 변환해서 담음
    dataY.append(char_to_int[seq_out])  # 정답을 담음

# print(dataX) # [[3, 2, 5, 5, 7, 0, 9, 7, 6, 0], ...
# print(dataY) # [1, 8, 2, 0, 11, 7, 10, 0, 7, 4]

N = len(dataX)  # 전체 학습 sample(시퀀스)갯수
print(f'전체 학습 sample갯수 : {N}')
if N == 0:
    raise ValueError("데이터가 적어 학습 시퀀스 생성 불가")

# x와y에 대한 one-hot encoding 처리
x = to_categorical(dataX, num_classes=n_vocab)
y = to_categorical(dataY, num_classes=n_vocab)

print(f"x shape : {x.shape} | y shape {y.shape}") # x shape : (10, 10, 12) | y shape : (10, 12)

# model정의하기
model = Sequential([
    Input(shape=(seq_length, n_vocab)),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(n_vocab, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)
print(model.summary())

chkpoint_path = 'data_stru/rnn6_model.keras'
os.makedirs(os.path.dirname(chkpoint_path), exist_ok=True)
chkpoint = ModelCheckpoint(
    chkpoint_path, monitor='loss', save_best_only=True, mode='min', verbose=0
)
early = EarlyStopping(
    monitor='loss', patience=10, restore_best_weights=True
)
batch_size = min(8, max(1, N // 2)) # 데이터 량에 따라 배치사이즈 조절 - 최소1, 최대 8

history = model.fit(
    x, y, epochs=500, batch_size=batch_size, callbacks=[chkpoint, early]
)

# 학습 곡선에 대한 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx() # loss와acc를 한그래프에 표시하기 위해 twinx 사용

loss_ax.plot(history.history['loss'], label='train loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], label='train accurcy')
acc_ax.set_ylabel('accurcy')
acc_ax.legend(loc='lower left')

plt.tight_layout()
plt.show()

# 모델이 예측한 확률분포(softmax사용하니까)에서 temperature와 top_k를 적용해 다른글자의 인덱스를 무작위로 선택
# temperature에서 결정함 : 모델이 예측한 확률을 그대로 쓰지 않고 조금 더 무작위성을 줌 
# sampling 함수 정의하기 =========================
def sample_with_temperatureFunc(probs, temperature=0.8, top_k=5):
    p = np.asarray(probs, dtype=np.float64) # 확률값 numpy배열로 변환

    # 상위 k개 확률만 남기기
    if top_k is not None and top_k > 0 and top_k < len(p):
        idx = np.argpartition(p, -top_k)[-top_k:]
        mask = np.zeros_like(p) # p와 동일한 구조의 0으로 채워진 배열 생성
        mask[idx] = p[idx] # 선택된 k개 위치만 원래 확률을 유지
        p = mask    # p값은 확률 백터를 상위 k개만 남긴 형태로 갱신 - 낮은 후보의 문자는 제외 
    
    # temperture 조절
    # 확률 -> log확률로 변환 softmax역변환하기 (1e-9 : log(0)방지)
    p = np.log(p + 1e-9) / max(temperature, 1e-8)
    p = np.exp(p) # 다시 지수확률(softmax)로 변환
    p = p / p.sum() # 확률 재 정규화, 확률의 총합이 1이 되도록 조정

    # 확률 p에 따라 인덱스 하나를 무작위로 선택된 인덱스를 정수로 반환 - 샘플링으로 인덱스 선택
    return int(np.random.choice(len(p), p=p)) 
''' --------------------------------------------
# np.argpartition(대상배열, 기준인덱스) : 부분 정렬 함수 (전체 정렬이 아니라 상위k개의 인덱스 반환)
k = 3
arr = np.array([7, 2, 9, 4, 1])
idx = np.argpartition(-arr, k)[:k]  # [2 0 3]
idx = np.argpartition(arr, k)[:k]   # [4 1 3]
--------------------------------------------'''
print()

# 문장 생성하기 =========================
start = np.random.randint(0, N - 1) # 랜덤 시작 인덱스 - 무작위로 시작
pattern = list(dataX[start])        # 시작 시퀀스
print(pattern) # [9, 7, 6, 0, 1, 8, 2, 0, 11, 7]
seed_text = "".join(int_to_char[v] for v in pattern)
print(f'"seed text" : \n{seed_text}\n') # tom are yo

steps = 500 # 생성할 문자 수
temperature = 0.8
top_k = 5

ganerated = [] # 생성결과 저장 리스트

for _ in range(steps):
    x = to_categorical([pattern], num_classes=n_vocab)
    probs = model.predict(x, verbose=0)[0] # 다음 문자의 확률 예측
    # 샘플링으로 문자 인덱스 선택
    idx = sample_with_temperatureFunc(probs, temperature=temperature, top_k=top_k)
    ch = int_to_char[idx]
    ganerated.append(ch)
    pattern.append(idx) # 입력 시퀀스 갱신(슬라이딩 윈도우)
    pattern = pattern[1:] # 시퀀스 슬라이딩을 맨앞글자를 제거하고 진행

gen_text = ''.join(ganerated) # 문자열 결합
print(gen_text)
