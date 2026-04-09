'''
분류 모델 성능파악(전부 사용가능하다.)
ROC Curve (Receiver Operating Characteristic) Curve
모든 분류 임계값에서 분류 모델의 성능을 보여주는 그래프로 
x축이 FPR(1-특이도), y축이 TPR(민감도)인 그래프
그래프 아래의 면적을 이용해 모델의 성능을 평가한다. AUC가 클 수록 정확히 분류함을 뜻한다
fpr(1-특이도: 위양성률)이 변할 때 tpr(민감도)이 어떻게 변하는지 알려주는 곡선이다.
'''
from sklearn.datasets import make_classification # 분류를 위해 데이터 만들어줌
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=2) 
# n_redundant : 독립변수 중 다른 독립변수의 선형조합으로 나타내는 성분 수
print(x[:3], x.shape) # (100, 2)
print(y[:3], y.shape) #  (100,)

# 산점도 그리기
# plt.scatter(x[:,0], x[:,1])
# plt.show()

model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print('y^ :', y_hat[:5])
print('y :', y[:5])

# Roc curve의 판별경계선 설정용 결정함수 사용
f_value = model.decision_function(x)
print('f_value : ',f_value[:10])
print()

df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f','y_hat','y'])
print(df.head())
print()

# 모델의 성능 파악
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat))
#     P         N
# T [[40(TP)    9(FN)]
# F [ 5 (FP)    46(TN)]]
acc = (40 + 46) / 100       # (TP + TN) / 전체수
recall = 40 / (40 + 9)      # TP / (TP + FN)
precission = 40 / (40 + 5)  # TP / (TP + FP)
specificity = 46 / (5 + 46) # TN / (FP + TN) - 특이도
fallout = 5 / (5 + 46)      # FP / (FP + TN) - 위양성율(fpr)
print('acc : ', acc)   
print('recall(TPR - roc curve y축, 1에 근사하면 좋다.) : ', recall)   
print('precission : ', precission)   
print('specificity : ', specificity)   
print('fallout(FPRate - roc curve x축, 0에 근사하면 좋다.)', fallout)   
print('fallout(== 1-specificity)', 1-specificity)   
print()

from sklearn import metrics
acc_score = metrics.accuracy_score(y, y_hat)
print("모델 정확도 :",acc_score)

cl_rep = metrics.classification_report(y, y_hat)
print('분류모델 리포트 :\n',cl_rep)
print()

# ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
# thresholds : 분류결정 입계값(결정함수값)
print('fpr :',fpr)
print('tpr :',tpr)

plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1],[0,1], 'k--', label='landom classifier line(AUC:0.5)')
plt.plot([fallout],[recall],'ro', ms=6) # fpr, tpr 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve')
plt.legend()
plt.show()

print('AUC(Area Under the Curve : ROC Curve의 면적 : 1에 근사할 수록 좋다. )출력')
print('AUC :',metrics.auc(fpr, tpr)) #  0.92797 - 매우 성능이 우수한 모델임을 알 수 있다.