import joblib as jb
import pandas as pd
import numpy as np
import os
import time

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])
val_data = val_data.reshape(val_data.shape[0], val_data.shape[1]*val_data.shape[2])

# 2. Model
# model = VAR(train_data,label_data).fit(verbose=2)
# model = RandomForestRegressor(random_state=1234, n_estimators=100, verbose=2)
model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234, n_estimators=100)


# 3. Compile, Fit
model.fit(train_data,label_data)


# model.save('D:\study_home\_save\_h5/vegi_ml01.h5')
# model = load_model('D:\study_home\_save\_h5/vegi.h5')


# 4. Evaluate, Predict
loss = model.score(val_data, val_target)
print(loss)

test_input = test_input.reshape(test_input.shape[0], test_input.shape[0] * test_input.shape[0])
test_pred = model.predict(test_input)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(val_target, test_pred)
# print('rmse: ', rmse)


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
with zipfile.ZipFile("submissionML.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()


