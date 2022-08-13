import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time

from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout


# 1. 데이터
filepath = 'D:/study_home/_data/dacon_antena/'
train = pd.read_csv(filepath+'train.csv', index_col=0)
test = pd.read_csv(filepath+'test.csv', index_col=0)

# print(train.head())
# print(train.info())
# print(train.isnull().sum())
# print(train.columns)

# 결측치 없음, 근데 뭐의 측정값의 0이 결측치란 얘기 있음
# x01~x56 / y01~y14
# 목표: output 14개 칼럼짜리 앙상블

x = train.drop(['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07',
       'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'], axis=1)

y_ = train.drop(['X_01', 'X_02', 'X_03', 'X_04', 'X_05', 'X_06', 'X_07', 'X_08', 'X_09',
       'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18',
       'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27',
       'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36',
       'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44', 'X_45',
       'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54',
       'X_55', 'X_56'],axis=1)

print(x.shape, y_.shape) # (39607, 56) (39607, 14)

y01 = train['Y_01']
y02 = train['Y_02']
y03 = train['Y_03']
y04 = train['Y_04']
y05 = train['Y_05']
y06 = train['Y_06']
y07 = train['Y_07']
y08 = train['Y_08']
y09 = train['Y_09']
y10 = train['Y_10']
y11 = train['Y_11']
y12 = train['Y_12']
y13 = train['Y_13']
y14 = train['Y_14']


y01 = np.array(y01).reshape(-1, 1)
y02 = np.array(y02).reshape(-1, 1)
y03 = np.array(y03).reshape(-1, 1)
y04 = np.array(y04).reshape(-1, 1)
y05 = np.array(y05).reshape(-1, 1)
y06 = np.array(y06).reshape(-1, 1)
y07 = np.array(y07).reshape(-1, 1)
y08 = np.array(y08).reshape(-1, 1)
y09 = np.array(y09).reshape(-1, 1)
y10 = np.array(y10).reshape(-1, 1)
y11 = np.array(y11).reshape(-1, 1)
y12 = np.array(y12).reshape(-1, 1)
y13 = np.array(y13).reshape(-1, 1)
y14 = np.array(y14).reshape(-1, 1)

x = np.array(x)
print(y14.shape, x.shape)

'''
# 2. 모델
input = Input(shape=(56,))
m = Dense(10, activation='relu')(input)
m = Dense(10, activation='relu')(m)

out1 = Dense(10)(m)
out2 = Dense(10)(m)
out3 = Dense(10)(m)
out4 = Dense(10)(m)
out5 = Dense(10)(m)
out6 = Dense(10)(m)
out7 = Dense(10)(m)
out8 = Dense(10)(m)
out9 = Dense(10)(m)
out10 = Dense(10)(m)
out11 = Dense(10)(m)
out12 = Dense(10)(m)
out13 = Dense(10)(m)
out14 = Dense(10)(m)

model = Model(inputs=input, outputs=[out1, out2 , out3 , out4 , out5 , out6 , out7 , out8 , out9 , out10, out11, out12, out13, out14])

# model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70, restore_best_weights=True)
log = model.fit(x, [y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14], epochs=1, batch_size=32, callbacks=[Es], validation_split=0.2)
'''
# model.save('D:\study_home\_save\_h5/antena.h5')
model = load_model('D:\study_home\_save\_h5/antena.h5')

# 4. 평가, 예측
# results = model.evaluate(x, [y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14])
# y_pred = model.predict(x)
# r2 = r2_score(y_pred, test)
# print('evaluate 결과: ', results)
# print('r2: ', r2)

# 5. 제출 준비
y_submit = model.predict(test)

print(y_submit)

submission = pd.read_csv(filepath+'sample_submission.csv', index_col=0)
submission = pd.DataFrame(y_submit, ['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07', 'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'])
submission.to_csv(filepath + 'submission.csv', index = True)
