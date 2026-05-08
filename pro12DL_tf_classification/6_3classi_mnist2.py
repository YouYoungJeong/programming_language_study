''' tf24
내가 그린 숫자 이미지를 mnist로 학습한 모델로 분류 예측하기
'''

# 손글씨 이미지 읽기
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
# TensorFlow oneDNN 최적화 로그/기능 끄기 - import tensorflow as tf보다 위에적용
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 이미지 불러오기
im = Image.open('num.png')
# 원본 이미지 크기 조절 + 
# convert('L') : L모드 이미지 흑백모드로 변경하고 픽셀값 0~255 범위(0:검정, 255:흰색)
img = np.array(im.resize((28, 28), Image.Resampling.LANCZOS).convert('L'))
print(img.shape) # (28, 28)

plt.imshow(img, cmap='Grays')
plt.show() # channel은 1개

# reshape , 정규화
data = img.reshape([1, 784]).astype('float32')
data /= 255.0

import tensorflow as tf
model = tf.keras.models.load_model('tf23model.keras')
new_pred = model.predict(data, verbose=0)
print(f'new pred : {new_pred}')
print(f'예측값 : {np.argmax(new_pred ,axis=1)[0]}')
