import numpy as np
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
import pandas as pd
from sklearn.metrics import classification_report

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show() 

def show_images_labels_predictions(images,labels,
                                  predictions,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        #顯示黑白圖片
        ax.imshow(images[start_id], cmap='binary')
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[start_id])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)') 
            title += '\nlabel = ' + str(labels[start_id])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[start_id])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()
    
def show_accuracy_loss(train_history):
    #accuracy準確率
    hist = pd.DataFrame(train_history.history)
    plt.figure(figsize=[8,8])
    plt.subplot(2,1,1)
    plt.plot(hist['accuracy'], 'r' , label='training')
    plt.plot(hist['val_accuracy'], 'b' , label='validate')
    plt.title('Accurary')
    plt.legend()

    #loss損失函數
    plt.figure(figsize=[8,8])
    plt.subplot(2,1,2)
    plt.plot(hist['loss'], 'r' , label='training')
    plt.plot(hist['val_loss'], 'b' , label='validate')
    plt.title('Loss')
    plt.legend()


#建立訓練資料和測試資料，包括訓練特徵集、訓練標籤和測試特徵集、測試標籤	
(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()

#show_image(train_feature[0]) 
#show_images_labels_predictions(train_feature,train_label,[],0,10)    

#將 Features 特徵值轉換為 784個 float 數字的 1 維向量
train_feature_vector =train_feature.reshape(len(train_feature), 784).astype('float32')
test_feature_vector = test_feature.reshape(len( test_feature), 784).astype('float32')

#print("train_feature_vector = ", train_feature_vector.shape)
#print("test_feature_vector = ", test_feature_vector.shape)
#train_feature_vector =  (60000, 784) 6萬張圖片，每一張皆轉為784個float數字的1為向量
#test_feature_vector =  (10000, 784)  1萬張圖片，每一張皆轉為784個float數字的1為向量

#Features 特徵值標準化
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

#label 轉換為 One-Hot Encoding 編碼
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

#建立模型
model = Sequential()
#輸入層：784, 隱藏層：256
model.add(Dense(units=256, 
                input_dim=784, 
                kernel_initializer='normal', 
                activation='relu')) 
# Dropout層防止過度擬合，斷開比例:0.3
model.add(Dropout(0.3))

#第二個隱藏層：128
model.add(Dense(units=128,  
                kernel_initializer='normal', 
                activation='relu')) 
# Dropout層防止過度擬合，斷開比例:0.3
model.add(Dropout(0.3))

#輸出層：10
model.add(Dense(units=10, 
                kernel_initializer='normal', 
                activation='softmax'))
#定義訓練方式
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

#以(train_feature_normalize,train_label_onehot)資料訓練，
#訓練資料保留 20% 作驗證,訓練10次、每批次讀取200筆資料，顯示簡易訓練過程
train_history =model.fit(x=train_feature_normalize,
                         y=train_label_onehot,validation_split=0.2, 
                         epochs=10, batch_size=200,verbose=2)

#評估準確率
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print('\n準確率=',scores[1])

#預測
#prediction=model.predict_classes(test_feature_normalize)
#'Sequential' object has no attribute 'predict_classes'
#更改下列網址程式碼
#https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
#model.predict_classes() is deprecated and will be removed after 2021-01-01. 
prediction = np.argmax(model.predict(test_feature_normalize), axis=-1)

#顯示圖像、預測值、真實值 
show_images_labels_predictions(test_feature,test_label,prediction,0)

#accuracy準確率 loss損失函數
show_accuracy_loss(train_history)

#confusion matrix
print("\n",pd.crosstab(test_label,prediction,rownames=['actual label'],colnames=['prediction']),"\n")

#顯示precision, recall, f1-score
print('classification')
print(classification_report(test_label,prediction))


