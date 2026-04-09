'''
랜덤 포레스트 (Random Forest) 분류 알고리즘
    랜덤 포레스트(Random Forest)는 수많은 의사결정 나무(Decision Tree)를 생성하고, 
    이들의 예측 결과를 다수결(분류)이나 평균(회귀)으로 종합하여 정확도를 높이는 
    대표적인 앙상블 학습 알고리즘. 데이터 무작위 샘플링(Bagging)과 피처 배깅을 통해 
    과적합을 방지하고 이상치에 강한 장점이 있다.
    
    앙상블 기법중 배깅(Bagging, Bootstrap aggregating)
        복수의 데이터 무작위 샘플링데이터와 수많은 의사결정 나무(Decision Tree)를 학습시키고 결과집계
        대표적인 알고리즘이 Random Foreset
    
    참고로 우수한 성능은 Boosting
    과적합이 걱정된다면 Bagging

        titanic dataset 사용하기
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv")
print(df.head(2))
print(df.info())
print(df.isnull().any())
print(df.shape)     # (891, 12)

# 관심있는 데이터 nan값 제거
df = df.dropna(subset=['Pclass','Age','Sex'])
print(df.shape)     # (714, 12)

# 관심있는 데이터 추출
df_x = df[['Pclass','Age','Sex']] # feature
print(df_x.head(3))
print()

# 데이터 전처리
# Label Encoding : 문자 범주형 데이터('Sex' col) 정수화 하기(dummy)
from  sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
df_x.loc[:, 'Sex'] = encoder.fit_transform(df_x['Sex']) # 최종 feature생성
print(df_x.head(3)) # 사전순으로 넘버링함:  female :0, male :1
df_y = df['Survived'] # label(class, target)
print(df_y.head(3))         # 사망:0 , 생존:1
print()

# train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (499, 3) (215, 3) (499,) (215,)
print()

# 모델 생성
model = RandomForestClassifier(criterion='gini', 
                            n_estimators=500, # n_estimators : 의사결정트리 수(실무 2000개정도 사용)
                            random_state=12)
model.fit(train_x, train_y)
pred = model.predict(test_x)
print('예측값 : ', pred[:5])            # [1 0 0 0 0]
print('실제값 : ', test_y[:5].values)   # [1 0 0 0 1]
print('맞춘 갯수 : ', len(test_y),'중',sum(test_y == pred),'개') # 215 중 178 개
print('전체 대비 맞춘 비율 : ', sum(test_y == pred)/len(test_y)) # 0.8279
print('분류 정확도 : ', accuracy_score(test_y, pred)) # 0.8279