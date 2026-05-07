'''
imdb dataset으로 이진 분류하기
    영화 리뷰 - 긍정(1) / 부정(0)
    train : 25000
    test  : 25000
'''
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, Input
# Embedding : 모든단어를 숫자화(인덱싱)시켜 패턴으로 학습시킴.
# Dropout : 딥러닝 학습 과정에서 일부 뉴런을 '무작위'로 비활성화하여 과적합(Overfitting)을 방지하는 정규화(Regularization) 기법
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os

# num_words=10000 : 자주 등장하는 단어 1만개만 사용
num_words=10000
(train_data, train_label),(test_data, test_label) = imdb.load_data(num_words=num_words)
print(type(train_data), train_data.shape)   # <class 'numpy.ndarray'> (25000,)
print(type(test_data), test_data.shape)     # <class 'numpy.ndarray'> (25000,)
print(train_data[0], len(train_data[0]))    # ... 19, 178, 32] 218
# 하나의 문장의 단어들이 인덱싱 되어있는 상태.
# 이미 전처리 된 데이터 - 각 리뷰(단어)가 숫자화 되어있다, 각 숫자는 고유단어 색인 -> 어휘사전 형태
print(train_label) # [1 0 0 ... 0 1 0] 긍정(1) / 부정(0)

# 참고로 이 리뷰 데이터 한개(0번째)를 원래 문장으로 보기 
word_index = imdb.get_word_index()
# print(word_index)
# 각 단어 인덱싱 확인
sorted_word_index = sorted(word_index.items(), key=lambda x:x[1])
for word, index in sorted_word_index[:10]:
    print(word, index)


reverse_word_index = {
    index + 3:word  # 특수 토큰 세개가 선행하므로 실제단어 index + 3
    for word, index in word_index.items()       
}
# 특수 토큰
reverse_word_index[0] = "<PAD>"     # 패딩
reverse_word_index[1] = "<START>"   # 문장의 시작
reverse_word_index[2] = "<UNK>"     # 모르는 단어
reverse_word_index[3] = "<UNUSED>"  # 사용 안하는 단어

# 복원하기
# 0번째 리뷰 문장으로 복원
decord_review = " ".join(
    reverse_word_index.get(i, "?") for i in train_data[0]
    # i에 해당하는 단어가 있으면 그 단어 반환, 단어가 없으면 "?" 반환
)
print("0번째 문장 decord_review : ",decord_review)
# load_data()안에서는 0~3번을 특수 토큰으로 쓰기 때문에 실제 리뷰 데이터에서 the는 4번이 된다.
print("0번째 라벨 train_label : ",train_label[0])

# 리뷰 길이 확인 - padding하기전 확인하기
review_len = [len(review) for review in train_data]
print('최소 길이 :',np.min(review_len))
print('최대 길이 :',np.max(review_len))
print('평균 길이 :',np.mean(review_len))
print('중앙값 :',np.median(review_len))

plt.figure(figsize=(8, 5))
plt.hist(review_len, bins=50)
plt.xlabel('리뷰 길이')
plt.ylabel('건수')
plt.grid(True)
plt.show()

# padding : 리뷰 문장 길이가 다름. 모델에 넣기 전에  길이를 맞춤.
# 각 리뷰를 최대 200 단어 index로 맞춤. 길면 앞부분 자르고, 짧으면 0을 채움
# 패턴이기 때문에 다 쓸 필요가 없다
maxlen = 200
x_train = pad_sequences(train_data, maxlen=maxlen)      # (25000, 200)
x_test = pad_sequences(test_data, maxlen=maxlen)        # (25000, 200)
y_train = np.array(train_label).astype(np.float32)      # (25000,)
y_test = np.array(test_label).astype(np.float32)        # (25000,)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print('패딩 된 1번째 : ',x_train[1]) # [   0    0    0    0    0    0    0    0    0    0    0   1   194 ...

# 모델 저장용 폴더 생성
MODEL_DIR = './imdb_model/'
if not os.path.exists(MODEL_DIR): # 폴더가 없으면 만들어
    os.makedirs(MODEL_DIR)

modelpath = './imdb_model/imdb_best.keras' # 예전에는 h5, hdfs : 빅데이터 라는 의미, keras는 keras를 쓰는걸 권장함

model = Sequential([
    Input(shape=(maxlen, )),
    Embedding(
        # Keras의 Embedding 층에서 input_dim은 모델이 처리할 고유한 단어의 개수(어휘 사전의 크기, Vocabulary Size)를 의미
        # 리뷰 1개가 단어번호 200개로 들어옴.
        input_dim = num_words, 
        
        # 임베딩 레이어(Embedding Layer)에서 각 입력 데이터(예: 단어)를 표현할 밀집 벡터(dense vector)의 차원 수
        # 단어 하나를 32개 실수로 표현함.
        # 밀집벡터화(Dense Vectorization) : 
        #   대부분의 차원이 0이 아닌 실수값으로 채워진 고정된 크기(주로 수백~수천 차원)의 낮은 차원 벡터
        #   실수 기반의 고정 크기에 실수 값으로 채움
        #   텍스트나 데이터의 의미를 수백~수천 차원의 실수 벡터로 조밀하게 표현하는 기술
        #   ex :[0.2, -0.1, 0.03, 0.5 ...]
        output_dim=32
    ),
    GlobalAveragePooling1D(),  # 200개의 단어 벡터를 평균내서 리뷰 전체를 하나의 32차원 벡터화, 이것이 리뷰의 전체 특징이 됨.
    Dense(units=32, activation='relu'),
    Dropout(0.3),
    Dense(units=16, activation='relu'),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])
print(model.summary()) # Total params: 321,601

model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='binary_crossentropy',
        metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_loss',
                        patience=3, 
                        restore_best_weights=True)
chkpoint = ModelCheckpoint(
                filepath=modelpath,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
)

history = model.fit(x_train, y_train, epochs=50, batch_size=512,
                    validation_split=0.2, callbacks=[early_stop, chkpoint], verbose=2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("테스트 평가 손실 : ",loss)
print("테스트 평가 정확도 : ",acc)

# loss, acc 시각화 하기
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()
print()
print()

# 저장된 모델 읽어 분류 예측하기
best_model = load_model(modelpath)
best_loss, best_acc = model.evaluate(x_test, y_test, verbose=0)
print('best_model 평가 손실 :', best_loss)
print('best_model 평가 정확도 :', best_acc)

# 기존 테이터 사용해 예측
new_data =x_test[:5]
new_label =y_test[:5]
pred_prob = best_model.predict(new_data, verbose=2)
pred_class = (pred_prob >= 0.5).astype(int).ravel()
print('예측 확률 : ',pred_prob.ravel())
print('예측 값 : ',pred_class)
print('실제 값 : ',new_label.astype(int))
print()

for i in range(5):
    result = '긍정' if pred_class[i] == 1 else '부정'
    real = '긍정' if new_label[i] == 1 else '부정'
    print(f'{i + 1}번 리뷰 예측 : {result}, 긍정 확률 {pred_prob[i][0]:.3f}')
    print(f'{i + 1}번 리뷰 실제 : {real}')
    print()