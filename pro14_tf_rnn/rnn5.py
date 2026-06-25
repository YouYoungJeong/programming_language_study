# 단어(공백으로 구분) 단위 자연어 생성 - 소설 토지 데이터 사용
import tensorflow as tf
import numpy as np
import re

path_to_file = tf.keras.utils.get_file(
    'toji.txt',
    'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/rnn_test_toji.txt'
)

with open(path_to_file, encoding='utf-8', errors='ignore') as obj:
    raw_text = obj.read()

print(raw_text[:100])
print('문자 수 : ', len(raw_text))  # 677125

# 정제 후 corpus 만들기
def clean_str(text:str) -> str:
    text = re.sub(r"[^가-힣0-9() \n]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# print(clean_str('abc가나다   _^&$12하하'))

cleaned = clean_str(raw_text)
# print(cleaned)
corpus = cleaned.replace("\n", " [NL] ")  # 줄바꿈을 토큰으로 처리하기 위해 특수문자 사용
# print(corpus)

# 토큰 처리  : 문자열 -> 단어분리 -> 단어사전 -> 정수번호로 변환
vectorizer = tf.keras.layers.TextVectorization(
    standardize = None,
    split = "whitespace",
    output_mode = "int",
    output_sequence_length = None,
    vocabulary = None
)

# 단어사전 생성
vectorizer.adapt(tf.data.Dataset.from_tensor_slices([corpus]))

vocab = vectorizer.get_vocabulary()
# print(vocab)
PAD, UNK = 0, 1
vocab_size = len(vocab)
print(f'어휘 수 : {vocab_size} (PAD={PAD}, UNK={UNK})')
print('샘플 어휘 : ', vocab[:20])

token_ids = vectorizer(tf.constant([corpus])).numpy()[0]   # 토큰id 시퀀스
print('토큰 수 : ', len(token_ids))   # 164150
print(token_ids)  # tf.Tensor([   51 51341  2059 ...    49  1590   275]
# print(vocab[51], ' ', vocab[51341], ' ', vocab[2059])

if len(token_ids) <= 50:
    raise ValueError('토큰 수가 너무 적어 작업 안함')

# 학습용 시퀀스
SEQ_LEN = 15   # 입력 길이 (과거 25개의 토큰을 보고 다음 토큰 예측)
BATCH = 32     # 배치 크기
BUFFER = 2000  # 셔플 버퍼

# tf.data.Dataset은 텐서플로우에서 고성능 데이터 입력 파이프라인을 구축하기 위한 핵심 API.
# 메모리 내 데이터를 슬라이싱하여 데이터셋을 생성하고, .map(), .batch(), .shuffle() 등으로
# 전처리한 뒤 반복 순회하여 사용

ds = tf.data.Dataset.from_tensor_slices(token_ids)  # 배열이나 리스트를 한 개씩 잘라서 Dataset으로 만듦
ds = ds.window(SEQ_LEN + 1, shift=1, drop_remainder=True) # 한 칸씩 우측으로 밀기
ds = ds.flat_map(lambda w:w.batch(SEQ_LEN + 1))  # 각 윈도우를 텐서로 수집
# Dataset 안의 각 원소를 다시 Dataset으로 바꾼 뒤 하나로 펼치기
# 1 -> [1, 11]
# 2 -> [2, 12]
# 3 -> [3, 13]  ===> 1 11 2 12 3 13

# 하나의 토큰 묶음(chunk)을 입력(x), 정답(y), 가중치(w)로 나누는 함수
def split_xyFunc(chunk):   # chunk : SEQ_LEN + 1
    x = chunk[:-1]   # 입력 (마지막 값 제외)
    y = chunk[1:]    # 정답 - 각 시점의 다음 토큰 예측
    w = tf.cast(tf.not_equal(y, PAD), tf.float32)   # 정답이 실제 토큰이면 1, 정답이 PAD면 0
    return x, y, w

# 파이프라인 구축
ds = (ds.map(split_xyFunc, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(BUFFER)
        .batch(BATCH, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))  # 학습 데이터 준비 병렬화

windows = len(token_ids) - SEQ_LEN    # 윈도우 크기
steps_per_epoch = min(100, max(1, windows // BATCH))   # 0 방지
print('steps_per_epoch : ', steps_per_epoch) # 2564

# 모델
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, )),
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(vocab_size)    # activation='softmax'
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # loss를 직접 계산
model.compile(optimizer='adam', loss=loss_fn,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) # 정답 라벨이 원핫 인코딩이 아닌 경우

model.summary()

# 모델이 예측한 점수인 logits를 확률로 바꾼 뒤, 그 확률에 따라 다음 토큰 하나를 추출하는 함수
def sample_from_logits(logits, temperature=1.0, top_k=0, forbid_ids=(0, 1)):
    logits = logits.astype(np.float64)   # 순서1 : logitsㄹ을 받음

    # forbid_ids : 생성하면 안되는 토큰
    # 순서2 :  PAD, UNK 같은 금지 토큰을 제외
    for tid in forbid_ids:
        if 0 <= tid < logits.size:
            logits[tid] = -np.inf

        if temperature <= 0:   # temperature로 확률 분포 조절. 0에 근사 보수적, 커지면 창의적
            temperature = 1e-8
        logits = logits / temperature

        if top_k:   # top_k가 있으면 상위 k개 후보만 남김
            k = min(int(top_k), logits.size)
            if k > 0 and k < logits.size:
                idxs = np.argpartition(-logits, k)[:k]
                # 배열에서 값 자체를 정렬하는 함수가 아니라,
                # 특정 기준 위치에 올 원소들의 “인덱스”를 빠르게 찾는 함수다.
                # 특히 상위 K개 / 하위 K개 인덱스를 찾을 때 많이 사용한다.
                mask = np.full_like(logits, -np.inf)
                mask[idxs] = logits[idxs]
                logits = mask

        logits = logits - np.max(logits)  # softmax로 확률을 만듦
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum == 0 or not np.isfinite(probs_sum):   # 전부 -inf가 되는 특이 케이스 방어
            probs = np.ones_like(probs) / probs.size
        else:
            probs /= probs_sum

        return np.random.choice(len(probs), p=probs)  # 확률에 따라 샘플 하나 선택

idx2tok = np.array(vocab, dtype=object)   # 단어 사전 리스트(vocab)를 numpy 배열로 반환

# 토큰 id를 사람이 읽을 수 있는 문장으로 변환 함수
def ids_to_text(ids):    # 예) ids:[2,3,5] -> ['사람','간다','나는']
    toks = idx2tok[ids]
    toks = [("\n" if t == "[NL]" else t) for t in toks]   # [NL] 토큰을 실제 개행(\n)으로 복원
    return " ".join(toks).replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")

# 사용자가 넣은 시작 문장을 바탕으로 학습된 모델이 뒤에 이어질 문장으로 자동 생성하는 함수
def generateFunc(seed_text, max_new_tokens = 80, temperature=0.9, top_k=30):
    seed = clean_str(seed_text).replace("\n", " [NL] ")
    seed_ids = vectorizer(tf.constant([seed])).numpy()[0].tolist()

    # context 길이를 SEQ_LEN으로 맞춤(왼쪽 PAD로 채움) -> RNN 입력 길이 고정
    context = [PAD] * max(0, SEQ_LEN - len(seed_ids)) + seed_ids[-SEQ_LEN:]

    out_ids = []   # 생성된 토큰 id 누적
    for _ in range(max_new_tokens):
        x = np.array(context, dtype=np.int32)[None, :]  # (1, SEQ_LEN) 배치화
        logits = model.predict(x, verbose=0)[0, -1]  # 마지막 시점의 로짓만 사용
        tid = sample_from_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            forbid_ids=(PAD, UNK))

        out_ids.append(tid)
        context = context[1:] + [tid]  # context에 방금 샘플 추가. 우측 이동

    text = ids_to_text(out_ids)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# 일정 주기로 샘플 출력 - 학습 진행 상태 확인용 클래스
class SamplerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):   # 매 epoch이 끝날 때 마다 자동 호출
        # 5에폭 마다. 마지막 에폭엔 항상 출력
        if epoch % 5 != 0 and epoch != (self.params.get('epochs', 1) - 1):
            return

        seed = "귀녀의 모습을 한번 쳐다보고 떠나려 했다."
        sample = generateFunc(seed, max_new_tokens=80, temperature=0.9, top_k=30)
        print("\n[샘플 생성:", epoch)
        print(seed + " " + sample[:500])

EPOCHS = 2   # 사실 많이 줘~~~
history = model.fit(ds,
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[SamplerCallback()],
                    verbose=2)

print('final loss : ', float(history.history['loss'][-1]))
print('final acc : ', float(history.history['sparse_categorical_accuracy'][-1]))

# 최종 테스트
seed = "귀녀의 모습을 한번 쳐다보고 떠나려 했다."
out = generateFunc(seed, max_new_tokens = 100, temperature=0.8, top_k=40)
print('최종 결과 : \n')
print(seed + " " + out)