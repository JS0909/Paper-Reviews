import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

x = train.filter(regex='X')
y = train.filter(regex='Y') 

print(x.shape, y.shape) # (39607, 56) (39607, 14)

# x = x.drop(['X_10', 'X_11'], axis=1) # 결측치 있는 칼럼 제거
# test = test.drop(['X_10', 'X_11'], axis=1)


# '''
# 2. 모델
from sklearn.pipeline import make_pipeline
xgb = MultiOutputRegressor(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
                   n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7))
model = make_pipeline(MinMaxScaler(), xgb)

# 3. 컴파일, 훈련
model.fit(x, y)
# '''

path = 'D:\study_home\_save\_dat/'
jb.dump(model, path + 'antena.dat')
# model = jb.load(path + 'antena.dat')

# 4. 평가, 예측
results = model.score(x, y)
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


# evaluate 결과:  0.43261472712661864

# 02 evaluate 결과:  0.43397159896467347

# 03 evaluate 결과:  0.28970594972140823