'''
앙상블(Ensemble) 모델 랜덤 포레스트 (Random Forest) 분류 알고리즘
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
print()

# 교차검증 (KFold)
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print('교차 검증 :',cross_vali)
# [0.75524476 0.8041958  0.81818182 0.83216783 0.83098592]
print('교차검증 평균 정확도 :',np.round(np.mean(cross_vali),5)) # 0.80816
print()

# 중요변수 확인하기 ====================================================================
# 특정 중요 변수 확인하기
# feature_importances_ : 각 특성이 기여한 정도(중요도)를 수치로 표현
# 값의 합은 1, 수치가 클 수록 해당변수가 불순도 감소에 더 많이 기여함.
print('중요 변수 확인하기---------------')
print("특성(변수) 중요도 : ", model.feature_importances_) # 예측에 기여한 정도를 수치화한다.
# [0.16172779 0.49842824 0.33984396]
print()

# 중요변수 시각화하기
import matplotlib.pyplot as plt
import koreanize_matplotlib
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.title("특정 변수를 대상으로 중요 변수 확인하기")
plt.xlabel('Fearture 중요도(importances score)')
plt.ylabel('Feartures')
plt.yticks(range(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

# 전체변수를 대상으로 중요 변수 확인하기
print(df.info())
# PassengerId, Name - Survived와 상관없는 변수
# Ticket, Cabin - 바로 가용 불가 (Encoding 필요)

df_imsi = df[['Pclass','Age','Sex','Fare','SibSp','Parch']] # 필요한 자료 추출
df_imsi.loc[:, 'Sex'] = encoder.fit_transform(df_imsi['Sex']) # 'Sex' encoding

train_x, test_x, train_y, test_y = train_test_split(
        df_imsi, df_y, test_size=0.3, random_state=12
        )
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

model2 = RandomForestClassifier(criterion='gini', 
                            n_estimators=500,
                            random_state=12)
model2.fit(train_x, train_y)
importances = model2.feature_importances_

# 컬럼명 + 중요도
feature_df = pd.DataFrame({
    'feature':df_imsi.columns,
    'importances':importances
}).sort_values(by='importances', ascending=False)
print(feature_df)

# 시각화
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.barplot(x='importances', y='feature', data=feature_df, orient='h')
plt.title("전체변수를 대상으로 중요 변수 확인하기")
plt.xlabel('Fearture 중요도(importances score)')
plt.ylabel('Feartures')
plt.tight_layout()
plt.show()
plt.close()