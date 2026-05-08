'''tf22
    숫자 이미지 출력해보기
'''

# 손글씨 이미지 읽기
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
im = Image.open('num.png')
# 원본 이미지 크기 조절 + 
# convert('L') : L모드 이미지 흑백모드로 변경하고 픽셀값 0~255 범위(0:검정, 255:흰색)
img = np.array(im.resize((28, 28), Image.Resampling.LANCZOS).convert('L'))
print(img.shape) # (28, 28)

plt.imshow(img, cmap='Grays')
plt.show() # channel은 1개

# 784열1열로 reshape
data = img.reshape([1, 784]).astype('float32')
print(data, data.shape) # (1, 784)

# 정규화 (또는 밀집벡터로 만들기)
data = data / 225.0
print(data)

# (1, 784)로 정규화 -> (28, 28)로 구조를 바꾼 후 시각화
plt.imshow(data.reshape(28, 28), cmap='Greys')
plt.title("reshape : (1, 784) -> (28, 28)")
plt.show()