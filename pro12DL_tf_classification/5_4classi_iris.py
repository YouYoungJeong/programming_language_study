'''
iris dataset으로 꽃 종류 분류기
    layer수 에 따른 모델 성능 비교 + ROC Curve까지 진행
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

iris = load_iris()
# print(iris.DESCR)
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

x = iris.data # feature
print(x[:2])
y = iris.target # label
print(y[:2])
names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label OneHot
onehot = OneHotEncoder(categories='auto')
print('전 :', y.shape) # 전 : (150,)
# y = onehot.fit_transform(y[:, np.newaxis]).toarray() # 차원 확대하기
y = onehot.fit_transform(y[:, None]).toarray() # 차원 확대하기
print('후 :', y.shape) # 후 : (150, 3)
print(y[:2])

# feature StandardScale
scailer = StandardScaler()
x_scaled = scailer.fit_transform(x)
print(x_scaled[:2]) # [[-0.90068117  1.01900435 -1.34022653 -1.3154443 ]

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42
)
print(x_train.shape, y_train.shape) # (105, 4) (105, 3)

n_feature = x_train.shape[1]
n_classes = y_train.shape[1]
print(n_feature, n_classes) # 4 3

# layer의 갯수가 다른 model 여러개 생성 함수 생성
def create_custom_model(input_dim, output_dim, out_nodes, n, model_name='model'):
    # print(input_dim, output_dim, out_nodes, n, model_name)
    
    # model을 생성하는 함수
    def create_model():
        model = Sequential(name=model_name)
        model.add(Input(shape=(input_dim, )))
        
        # 은닉층 n개 생성
        for _ in range(n):
            model.add(Dense(units=out_nodes, activation='relu'))
        
        # 출력층 생성
        model.add(Dense(units=output_dim, activation='softmax'))

        # compile
        model.compile(
            loss = 'categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )
        return model
    
    # ▼▼▼ 클로저 : 함수를 실행하는게 아니라 함수의 주소를 넘김 - 실행X ▼▼▼ 
    #  클로저는 함수안의 내부함수로서 외부함수의 변수를 참조 할 수있는 놈
    return create_model 

# 모델 주소 저장
models = [create_custom_model(n_feature, n_classes, 10, n, "model_{}".format(n)) 
            for n in range(1, 4)] 

# print(models) # 함수의 주소가 넘어옴
# [<function create_custom_model.<locals>.create_model at 0x000001DE3F65E840>, 
#  <function create_custom_model.<locals>.create_model at 0x000001DE3F65E700>, 
#  <function create_custom_model.<locals>.create_model at 0x000001DE3F65E660>]

# 모델의 구조 확인하기
for create_model in models:
    print()
    create_model().summary()

history_dict = {}
for create_model in models:
    
    # 모델 실행
    model = create_model() 
    print(f'모델명 : {model.name}')
    historys = model.fit(
        x_train, y_train, batch_size=4, epochs=50, verbose=0, validation_split=0.3)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'loss:{score[0]:.4f}, acc:{score[1]:.4f}')
    
    # dict에 저장하기
    history_dict[model.name] = [historys, model]

print(history_dict)
# {'model_1': [<keras.src.callbacks.history.History object at 0x00000241D0E22CF0>, 
#               <Sequential name=model_1, built=True>], 
#  'model_2': [<keras.src.callbacks.history.History object at 0x00000241D0EB2990>, 
#               <Sequential name=model_2, built=True>], 
#  'model_3': [<keras.src.callbacks.history.History object at 0x00000241D0EB0410>, 
#               <Sequential name=model_3, built=True>]}

# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
for model_name in history_dict:
    # acc, loss 확인하기
    # print('h_d :', history_dict[model.name][0].history['acc'])
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('val acc')
    ax2.set_ylabel('val loss')
    ax2.set_xlabel('epoch')
    ax1.legend()
    ax2.legend()

plt.show()

# ROC Curve - 분류기에 대한 성능 평가 기법중 하나
plt.figure()
plt.plot([0, 1],[0, 1], 'k--')
for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())

    print(model_name," auc :",auc(fpr, tpr))
    plt.plot(fpr, tpr, label='{}, AUC:{:.3f}'.format(model_name, auc(fpr, tpr)))

plt.xlabel("fpr(false prositive rate)")    
plt.xlabel("tpr(true prositive rate)")
plt.title('ROC Curve')
plt.legend()    
plt.show()