'''ex41
SVM (Support Vector Machine) - PCA를 이용한 이미지 분류
    CNN워밍업
세계 정치인들 중 일부 얼굴사진 데이터를 사용.
LFW (Labeled Faces in the Wild) dataset 사용
    유명 정치인 등의 실제 얼굴에 대한 이미지 데이터. fetch_lfw_people() 명령으로 로드한다. 
    해상도는 50x37=5,828픽셀이고 각 채널이 0~255의 정수값을 가지는 컬러 이미지이다. 
    5,749명의 13,233개 사진을 가지고 있다. 
    다음 인수를 사용하여 로드하는 데이터의 양과 종류를 결정할 수 있다.

        funneled : 디폴트 True
        이미지의 위치 조정(funnel) 여부
        resize : 디폴트 0.5
        원래 이미지보다 일정 비율로 크기를 줄여서 로드
        min_faces_per_person : 디폴트 None
        각 인물당 최소 사진의 수
        color : 디폴트 False
        True인 경우에는 컬러 사진을, False인 경우에는 흑백 사진을 로드한다.
'''
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# np.set_printoptions(threshold=np.inf)

faces = fetch_lfw_people(min_faces_per_person=60, color=False, resize=0.5) # 전부 기본값
# 60 : 한사람당 60장 이상의 사진이 있는 자료만 사용 - 너무 작은 sample은 사용하지 않겠다는 뜻
# print(faces) # 이미지를 전부 숫자화해 패턴을 나타냄.
print(faces.DESCR) # data 설명 보기
print(faces.data)
print(faces.data.shape) 
print(faces.target)
print(faces.target_names)
# target = ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
# 'Gerhard Schroeder', 'Hugo Chavez', 'Junichiro Koizumi','Tony Blair']
print(faces.images.shape)   # (1348, 62, 47)
print()

# 이미지 1개 시각화
# print(faces.images[1])
# print(faces.target_names[faces.target[1]])
# plt.imshow(faces.images[1], cmap='bone')
# plt.show() # 62행 47열값(총 2914개)에 좌표값들에 점이 찍혀있다.

# 원본이미지 15개 시각화
fig, ax = plt.subplots(3, 5, figsize = (10, 6))
for i , axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]]) # 축은 없애고 label만 보기
plt.show() 

# 주성분 분석으로 이미지 차원을 축소시켜 분류 작업을 진행--------------------------
# 설명력 95% 되는 최소 개수를 얻기
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(faces.data)
print('설명력 95% 되는 최소 개수를 얻기 :',pca.n_components_) # 184

# 원본 데이터를 설명하는 부분은 제 1~2 주성분에 대부분 있기 때문에 갯수 조절에 의미가 없을 수 있다.
# 차원수(n)는 분석가가 판단한다 : 성능 확인 후 차원 수를 조절 한다
n = 150 
m_pca = PCA(n_components= n,
            whiten=True,    # 주성분의 스케일이 작아지도록 조정
            random_state=0)
x_low = m_pca.fit_transform(faces.data) # (1348, 2914) -> # (1348, n)
print(f"x_low : \n{x_low},\n{x_low.shape}") # (1348, n만큼) -> 열의 갯수가 n만큼 줄었다.

# 주성분 이미지 15개 시각화
fig, ax = plt.subplots(3, 5, figsize = (10, 6))
for i , axi in enumerate(ax.flat):
    axi.imshow(m_pca.components_[i].reshape(faces.images[0].shape), cmap='bone')
    # faces.images[0].shape() : [2914] 를
    # .reshape(faces.images[0].shape) : [62, 47]로 reshape함.
    axi.axis('off')
    axi.set_title(f'PC {i + 1}')
plt.suptitle("Eigenfaces(주성분 얼굴)", fontsize=12)
plt.tight_layout() 
plt.show() 
'''
출력 이미지는 실제 이미지가 아니라 특징 패턴을 보여줌
    특징 패턴 : (눈위치, 코, 그림자, 얼굴 윤곽 등등..)
SVM Algorithm은 실제 얼굴이 아니라 특징패턴으로 분류 작업을 진행함.
'''
# 설명력 확인하기
print("-"*20,"설명력 확인하기","-"*20)
print(m_pca.explained_variance_ratio_[:10])
print("누적 설명력 :",m_pca.explained_variance_ratio_.sum()) 
# n = 100 : 0.9039658
# n = 50 : 0.8351595
# n 개로 얼마나 원본 정보를 유지했는지 확인함

# 복원된 이미지 확인하기 (원본 vs 복원)
x_reconst = m_pca.inverse_transform(x_low)
fig, ax = plt.subplots(2, 5, figsize = (10, 4))
for i in range(5):
    # 원본
    ax[0, i].imshow(faces.images[i], cmap='bone')
    ax[0, i].set_title("원본")
    ax[0, i].axis('off')

    # 복원
    ax[1, i].imshow(
            x_reconst[i].reshape(faces.images[0].shape),
            cmap = 'bone'
            )
    ax[1, i].set_title("복원")
    ax[1, i].axis('off')

plt.suptitle("PCA 복원 비교", fontsize=10)
plt.tight_layout()
plt.show()
# 컴퓨터의 입장 : 원본 데이터와 복원된 이미지의 기본 특징은 크게 차이가 없다.(패턴이 유지됨)
print()

# 분류 모델 생성 (SVM)
svcmodel = SVC(C = 1, random_state=1)
mymodel = make_pipeline(
    m_pca,      # PCA와
    svcmodel,   # SVM분류기를 파이프라인으로 묶어서 순차적으로 실행
)
print('mymodel : ',mymodel)
# Pipeline(steps=[('pca', PCA(n_components=100, random_state=0, whiten=True)),
#                 ('svc', SVC(C=1, random_state=1))])
print()

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, 
                                                    faces.target, 
                                                    random_state=1,
                                                    stratify=faces.target, # 불균형 자료 완화
                                                    )
print(x_train.shape, x_test.shape) # (1011, 2914) (337, 2914)
print(x_train[0]) # [0.04052288 0.03006536 0.09803922 ... 0.76732033 0.7660131  0.7921569 ]
print(y_train[0]) # 3 <- George W Bush
print()

# fit model
mymodel.fit(x_train, y_train)
spred = mymodel.predict(x_test)
print(f"예측값 : {spred[:10]}")     # [3 3 3 3 3 0 2 3 3 1]
print(f"실제값 : {y_test[:10]}")    # [3 5 4 2 4 0 6 3 3 1]
print()

# 정확도
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confmat = confusion_matrix(y_test, spred)
print("confusion_matrix :\n",confmat)
# [[ 11   1   0   7   0   0   0   0]
#  [  0  50   0   9   0   0   0   0]
#  [  0   1  14  15   0   0   0   0]
#  [  0   0   0 133   0   0   0   0]
#  [  0   0   0  16  11   0   0   0]
#  [  0   1   0  11   0   6   0   0]
#  [  0   0   1   3   0   0  11   0]
#  [  0   0   0  12   0   0   0  24]]

print("accuracy_score :" ,accuracy_score(y_test, spred)) #  0.7715133
print("classification_report :\n",classification_report(y_test, spred, target_names=faces.target_names))

#                     precision  recall   f1-score   support
#     Ariel Sharon      1.00      0.58      0.73        19
#     Colin Powell      0.94      0.85      0.89        59
# Donald Rumsfeld       0.93      0.47      0.62        30
#     George W Bush     0.65      1.00      0.78       133
# Gerhard Schroeder     1.00      0.41      0.58        27
#     Hugo Chavez       1.00      0.33      0.50        18
# Junichiro Koizumi     1.00      0.73      0.85        15
#     Tony Blair        1.00      0.67      0.80        36
#         accuracy                           0.77       337
#         macro avg      0.94      0.63      0.72       337
#     weighted avg       0.84      0.77      0.76       337

# 분류 결과에 대한 시각화 ->  x_test[0]번째 1개만 보기
print('분류 결과 x_test[0]번째 시각화 하기')
plt.subplots(1, 1)
plt.imshow(x_test[0].reshape(62, 47), cmap='bone') # 1차원을 2차원으로 차원 변환
plt.show()

# 분류 결과에 대한 시각화 -> 24개 보기
print('분류 결과 24개 시각화 하기')
fig, axes = plt.subplots(4, 6)
for i, ax in enumerate(axes.flat):
    ax.imshow(
        x_test[i].reshape(62, 47), 
        cmap='bone'
    )
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel(faces.target_names[spred[i]].split()[-1],
                color='blue' if spred[i] == y_test[i] else 'red', # last name만 추출
                fontweight='bold')
fig.suptitle('예측 결과', fontsize = 12)
plt.tight_layout()
plt.show()

# 오차 행렬(못맞춘 데이터) 시각화하기
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confmat,            # confusion_matrix
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels= faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel("실제") 
plt.ylabel("예측") 
plt.title("confusion_matrix")
plt.show()

# PCA 누적 분산 그래프 (n_components=n값이 어떻게 나왔는지 확인하기)
plt.plot(np.cumsum(m_pca.explained_variance_ratio_))
plt.xlabel('주성분 개수')
plt.ylabel('누적 설명력')
plt.title('PCA 설명력')
plt.grid(True)
plt.show()
print()

# 새로운 이미지를 입력해 분류하기
# 현재 모델의 분류 accuracy : 0.77151
# 실습 1 - 기존데이터로 테스트
print("기존데이터로 이미지를 입력해 분류하기")
test_img = faces.data[0].reshape(1, -1) 
# 모델이 이형태로 학습을 했기 때문에 reshape - (1, 2914)형태로 변환을 필수로 해줘야한다.  
print("test_img : ",test_img)
test_pred = mymodel.predict(test_img)
print("실습1 예측결과(기존데이터로 예측하기) :", faces.target_names[test_pred[0]],\
        ", index:",test_pred[0])
print("실제값 :",faces.target_names[faces.target[0]],", index:",faces.target[0])
# 실습1 예측결과(기존데이터로 예측하기) : Colin Powell , index: 1
# 실제값 : Colin Powell , index: 1
print()

# 실습 2 - 새로운 데이터로 분류 작업하기 (이게 진짜~~~)
# 단계 : 이미지 읽기 -> 흑백 변환 -> 크기 맞추기(62 x 47) -> 일차원으로 변환 -> 예측
print("새로운데이터로 이미지를 입력해 분류하기")
from PIL import Image
img = Image.open("bush.jpg")    # 이미지 읽기
img = img.convert("L")          # 흑백 변환하기
img = img.resize((47, 62))      # 크기 맞추기 (width, height) 
'''
PIL.Image(width:가로, height:세로)는 numpy(height:세로, width:가로)랑 반대로 되어 있다.
=>이미지는 라이브러리마다 축 순서가 다 다르기 때문에 꼭 찾아봐야 한다.
'''
img_np = np.array(img)          # 사이즈 조절위해 numpy array로 형변환
# print("img_np :",img_np)      # 원본 데이터와 형태 맞춰주기 이미지는 대부분 정규화를 진행함.
'''
원본 데이터[[0.53333336 0.52418303 0.49673203 ... 0.00653595 0.00653595 0.00130719]]
새로운 이미지 데이터 [[52 51 50 ... 72 74 75]
'''
img_np = img_np / 255.0         # 정규화(Normalizing) 진행
# print("img_np :",img_np)      # [[0.20392157 0.2        0.19607843 ... 0.28235294 0.29019608 0.29411765]
img_flat = img_np.reshape(1, -1)# 1차원으로 변환

# 예측하기
new_pred = mymodel.predict(img_flat)
print("실습2 예측결과(새로운 데이터로 예측하기) :", faces.target_names[new_pred[0]],\
        ", index:",new_pred[0])

# 시각화 + 예측
plt.imshow(img_np, cmap='bone')
plt.title(f"예측 {faces.target_names[new_pred[0]]}")
plt.axis('off')
plt.show()
# 참고 : 정확도를 높이려면 밝기/위치 정렬등의 작업이 필요!!!

