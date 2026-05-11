'''tf26
합성곱의 원리 이해  - 컵의 특징 추출
(필터, 스트라이드, 피쳐맵 , 패딩, 플래튼(fc layer)) = 컨볼루션 , 풀링 
CNN의 알고리즘이 아주 훌륭함.
합성곱(Convolution)
    원본 이미지와 영상의 패턴을 추출할 수 있는 필터(Filter)를 이용하여 특징을 추출하는 과정이다. 
    필터는 원본 이미지를 움직이면서(Stride) 이미지의 특징을 뽑아내는 결과물(Feature Map)을 만든다. 
    결과물(Feature Map)은 원본 이미지의 인접한 픽셀 간 연관성 있는 패턴 정보를 잃지 않고 반영할 수 있다.
풀링(pooling)
    합성곱 연산을 통해 나온 결과물에서 대푯값들만 뽑아 내는 과정이다. 
    이미지 패턴 정보를 단순화, 추상화 하는 작업으로 생각할 수 있다. 
    풀링의 종류에는 최대(Max), 최소(Min), 평균(Average) 등 여러 가지가 있는데, 
    일반적으로 최대 풀링(Max-Pooling)을 사용한다.
필터(Filter)
    원본 입력 데이터에 대해 특징값을 뽑기 위해 만들어진 장치이다. 
    원본이미지와 필터를 합성곱 연산을 시키면 오른쪽 그림과 같이 다양한 특징, 
    혹은 관점으로 이미지를 인식할 수 있는 결과물이 나온다. 
    합성곱 신경망에서는 이 필터에 값들이 가중치로서 학습과정에서 데이터에 맞게 변경된다.
이미지 분류(Image Classification)
    입력으로 이미지 정보를 받아서 이미지가 어디에 속할지 분류하는 문제이다. 
    가령 숫자를 인식하는 문제나 주어진 사진이 개인지 고양이인지 분류하는 문제의 유형이 여기에 해당된다.   
    
    -> 컨볼루션 + 풀링 = CNN
    -> 원본이미지가 너무 고해상도 인경우 
    -> 컴퓨터 입장 메모리를 너무 많이 잡아먹어(Dense가 처리하는데 굉장히 힘듦)
    -> 이미지의 크기를 줄여야해서 filter를 가져다댐 : 컨볼루션
    -> 패딩을 하지 않으면 원본크기와 다른 이미지가 됨.(** open cv 의 내용에 나오는 내용)
    -> Feature map을 Dense의 진행하기위해 플래튼(1차원)을 진행 = FC layer -> 데이터의 패턴을 찾아 학습함.
    -> CNN은 이미지 뿐만 아니라 text도 진행할 수 있다.
    -> 결국은 Dense가 모든일을 처리하는건데 Dense위에 CNN을 얹은것.
    CNN(컨볼루션 신경망) : https://cafe.daum.net/flowlife/S2Ul/3
'''
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize

# 초기 컵 이미지를 읽어와 (64, 64)로 리사이즈
im = rgb2gray(data.coffee())
im = resize(im, (64, 64))
print(im.shape) # (64, 64)
print(im) # 이미지의 데이터값

plt.axis('off')
plt.imshow(im, cmap='gray')
plt.show()

# 합성곱 필터 ( 3 X 3 ) - 필터값에 따라 원본이미지 특징을 잡아줌 : CNN이 알아서 잡아줌  
# filter = np.array([
#     [1, 1, 1],
#     [0, 0, 0],
#     [-1, -1, -1]
# ])
filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# padding : 원본 이미지 외각 상하좌우에 1픽셀씩 0으로 채우기
new_image = np.zeros(im.shape)      # 64X64의 이미지 생성 - Feature Map
im_pad = np.pad(im, 1, 'constant')  # 1px씩 0으로 채우기

# 합성곱(원소별 곱의 합) 연산(Convolution)을 수행
# 원래 이미지 im의 크기에 대해 모든 px 좌표(i, j)를 훑는다.
for i in range(im.shape[0]):    # 0 ~ 63(세로방향)
    for j in range(im.shape[0]):# 0 ~ 63(가로방향)
        try:
            new_image[i, j] = \
                    im_pad[i - 1, j - 1] * filter[0, 0] + \
                    im_pad[i - 1, j] * filter[0, 1] + \
                    im_pad[i - 1, j + 1] * filter[0, 2] + \
                    im_pad[i , j - 1] * filter[1, 0] + \
                    im_pad[i , j] * filter[1, 1] + \
                    im_pad[i , j + 1] * filter[1, 2] + \
                    im_pad[i + 1 , j - 1] * filter[2, 0] + \
                    im_pad[i + 1 , j] * filter[2, 1] + \
                    im_pad[i + 1, j + 1] * filter[2, 2]                    
        except:
            pass

print(new_image) # 이미지의 합성곱 값 - Convolution

plt.axis('off')
plt.imshow(new_image, cmap='gray')
plt.show()
