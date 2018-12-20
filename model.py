import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model():
  #モデルの実装
  model = Sequential()

  model.add(Dense(50, activation='relu', input_shape=(11,)))
  model.add(Dropout(0.2))

  model.add(Dense(50, activation='relu', input_shape=(11,)))
  model.add(Dropout(0.2))

  model.add(Dense(50, activation='relu', input_shape=(11,)))
  model.add(Dropout(0.2))

  model.add(Dense(10, activation='softmax'))
  model.summary()

  return model
