import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.optimizers import Adam
from model import build_model
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


# modelのビルド
model = build_model()
model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['accuracy'])

#CSVファイルの読み込み
wine_data_set = pd.read_csv("data/Quality_preGame_DataSet.csv",header=0)

#説明変数(ワインに含まれる成分)
x = DataFrame(wine_data_set.drop(["No.","Quality"],axis=1))

#目的変数(各ワインの品質を10段階評価したもの)
y = DataFrame(wine_data_set["Quality"])

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,batch_size=128,epochs=1000,verbose=1,validation_data=(x_test, y_test))

model.save("model/model.h5")

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)

print("Test loss:",score[0])
print("Test accuracy:",score[1])

# 学習履歴をプロット
plot_history(history)
