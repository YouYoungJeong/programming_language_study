'''
지도학습    : 최근접 이웃(K-Nearest Neighbors)
비지도학습  : 비계층적 군집분석(Non-hierachical Clustering) - K-means
iris data로 KNN / KMean 비교하기 
'''
import numpy as np
from sklearn.datasets import load_iris
iris_data = load_iris()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(
    iris_data['data'],  
    iris_data['target'],
    test_size=0.25,
    random_state=42,
)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 
# (112, 4) (38, 4) (112,) (38,)
print()

print('지도학습(Supervised Learning) - KNN')
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
knnmodel.fit(train_x, train_y)

# 예측및 성능확인(acc)
from sklearn.metrics import accuracy_score
pred_label = knnmodel.predict(test_x)
print('예측값 :',pred_label[:10])   # [1 0 2 1 1 0 1 2 1 1]
print('실제값 :',test_y[:10])       # [1 0 2 1 1 0 1 2 1 1]
print('분류 정확도 :', accuracy_score(test_y, pred_label)) # 1.0
print()

# 새로운 값 분류
new_input =np.array([[6.1, 2.8, 4.7, 1.2]])
print("새로운값 예측하기(KNN)-label값 :",knnmodel.predict(new_input)) # [1]
print()

# 새로운 데이터는 몇번 째 자료와 거리를 확인했을까?
dist, index = knnmodel.kneighbors(new_input)
print(dist, index) 
# [[0.2236068  0.3     0.43588989]] [[71 82 31]] 
# k=3 이므로 [[71 82 31]] 번 자료가 분류에 참여함 => setosa:1 < versicolor:2  (versicolor로 예측됨) 
# 새로운 데이터에서 각 참여한 데이터의 거리가 [[0.2236068  0.3     0.43588989]]
print()

# =================================================================================
print('비지도학습(Unsupervised Learning) - KMeans') 
from sklearn.cluster import KMeans
# kmeans - Warning 무시
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

kmeansModel = KMeans(n_clusters=3, init='k-means++',random_state=0)
kmeansModel.fit(train_x) # Label이 주어지지 않음.
print("군집의 라벨 출력 :",kmeansModel.labels_) # 군집의 실제 라벨 출력
print()

# 군집별 자료보기
print('0 cluster :',train_y[kmeansModel.labels_== 0]) # 0번째 군집은 label=2
print('1 cluster :',train_y[kmeansModel.labels_== 1]) # 1번째 군집은 label=0
print('2 cluster :',train_y[kmeansModel.labels_== 2]) # 2번째 군집은 label=1
# 0 cluster : [2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2]
# 1 cluster : [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# 2 cluster : [2 1 1 1 2 1 1 1 1 1 2 1 1 1 2 2 2 1 1 1 1 1 2 1 1 1 1 2 1 1 1 2 1 1 1 1 1
#  1 1 1 1 1 1 2 2 1 2 1]
print()

# 새로운 값 군집 분류
new_input =np.array([[6.1, 2.8, 4.7, 1.2]])
clu_pred = kmeansModel.predict(new_input)
print("새로운값 예측하기(KMeans)-군집값 :",clu_pred) # 군집값[2] == label=1
print()

# 군집 모델 성능 파악하기
pred_cluster = kmeansModel.predict(test_x)
print("예측값 :",pred_cluster) #  [2 1 0 2 2...
print()

# 성능 비교를 위해 평가데이터를 적용해 예측한 군집을 각 
# iris의 종류를 의미하는 라벨값으로 다시 바꿔줘야 실제 라벨과 비교해서 성능 측정 가능
# 값을 미리 알고 있어서 가능함.
np_arr = np.array(pred_cluster)
print('np_arr : ', np_arr)
np_arr[np_arr == 0], np_arr[np_arr == 1], np_arr[np_arr == 2]= 3, 4, 5 # 임시 저장용
np_arr[np_arr == 3] = 2 # 군집3을 2(versicolor)로 분류
np_arr[np_arr == 4] = 0 # 군집4을 0(setosa)로 분류
np_arr[np_arr == 5] = 1 # 군집5을 1(verginicar)로 분류
print(np_arr)
predict_label = np_arr.tolist()
print('군집 test acc :', np.mean(predict_label==test_y)) # 0.947368