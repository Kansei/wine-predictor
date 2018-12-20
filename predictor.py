import keras
from keras.models import load_model
import pandas as pd
from pandas import DataFrame
import sys

def restore_model():
    return load_model('model/model.h5')

argvs = sys.argv
argc = len(argvs) # 引数の個数
if (argc != 2):
    print('Usage: # python %s input-data' % argvs[0])
    quit()

#CSVファイルの読み込み
wine_data_set = pd.read_csv(argvs[1],header=0)

#説明変数(ワインに含まれる成分)
x = DataFrame(wine_data_set.drop("No.",axis=1))

model = restore_model()

predict = model.predict_classes(x,batch_size=1,verbose=0)

print(predict)
