import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time
import joblib as jb

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

x = x.drop(['X_10', 'X_11'], axis=1) # 결측치 있는 칼럼 제거
test = test.drop(['X_10', 'X_11'], axis=1)

# '''
# 2. 모델
model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234)

# 3. 컴파일, 훈련
model.fit(x, y_)
# '''

path = 'D:\study_home\_save\_dat/'
jb.dump(model, path + 'antena.dat')
# model = jb.load(path + 'antena.dat')

# 4. 평가, 예측
results = model.score(x, y_)
# y_pred = model.predict(x)
# r2 = r2_score(y_pred, test)
print('evaluate 결과: ', results)
# print('r2: ', r2)

# 5. 제출 준비
y_submit = model.predict(test)
submission = pd.read_csv(filepath+'sample_submission.csv', index_col=0)

for idx, col in enumerate(submission.columns):
    if col=='ID':
        continue
    submission[col] = y_submit[:,idx-1]
    
submission.to_csv(filepath + 'submission.csv', index=True)


# evaluate 결과:  0.43218261725515184

# evaluate 결과:  0.43261472712661864