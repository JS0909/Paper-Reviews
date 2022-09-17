import joblib as jb
import pandas as pd
import numpy as np
import os
import time

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM, Dropout, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Data
path = 'D:\study_home\_data\dacon_vegi/'

train_data, label_data, val_data, val_target, test_input, test_target = jb.load(path+'datasets.dat')

# print(train_data[0])
# print(len(train_data), len(label_data)) # 1607 1607
# print(len(train_data[0]))   # 1440
# print(label_data)   # 1440
# print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
# print(val_data.shape) # (206, 1440, 37)
# print(test_target.shape) # (195,)

val_data = (val_data - np.mean(train_data)) / np.std(train_data)
train_data = (train_data - np.mean(train_data)) / np.std(train_data)


# 2. Model
model = Sequential()
# model.add(GRU(256, input_shape=(1440,37)))
model.add(Bidirectional(LSTM(100, input_shape=(1440,37))))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# model.add(GRU(100,input_shape=(1440,37)))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))


# 3. Compile, Fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)

start = time.time()
hist = model.fit(train_data,label_data, batch_size=2000, epochs=50, callbacks=[Es], validation_data=(val_data, val_target))
end = time.time()

model.save('D:\study_home\_save\_h5/vegi08.h5')
# model = load_model('D:\study_home\_save\_h5/vegi08.h5')


# 4. Evaluate, Predict
('시간:', end-start, '\n')
print(model.evaluate(val_data, val_target))

test_pred = model.predict(test_input)

# test_pred -> TEST_ files
for i in range(6):
    thislen=0
    thisfile = 'D:\study_home\_data\dacon_vegi/test_target/'+'TEST_0'+str(i+1)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])


# TEST_ files -> zip file
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_home\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submissionKeras.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()


# vegi01
# [0.2958856523036957, 0.2879069149494171]

# vegi02
# [0.290480375289917, 0.26576611399650574]

# vegi03
# [0.29135653376579285, 0.24336576461791992]

# vegi04 시간: 840.793340921402 / 학원
# [0.28843000531196594, 0.26544904708862305]

# vegi05 시간: 3190.4104120731354
# [0.2806214988231659, 0.25756171345710754]

# vegi06
# [0.28508618474006653, 0.24901825189590454]

# vegi07
# [0.2825902998447418, 0.24616381525993347]