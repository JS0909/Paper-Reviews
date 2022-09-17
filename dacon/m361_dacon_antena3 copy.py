import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from catboost import CatBoostRegressor



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(999) # Seed 고정

filepath = 'D:/study_home/_data/dacon_antena/'

train = pd.read_csv(filepath + 'train.csv',index_col=0)
test = pd.read_csv(filepath + 'test.csv').drop(columns=['ID'])

train_x = train.filter(regex='X') # Input : X Featrue
train_y = train.filter(regex='Y') # Output : Y Feature

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=42, shuffle=True)

for train_index, test_index in kf.split(train_x):
    train_xx, valid_xx = train_x.iloc[train_index], train_x.iloc[test_index]
    train_yy, valid_yy = train_y.iloc[train_index], train_y.iloc[test_index]
    break

new_train_xx = train_xx.drop(columns = "X_04")
new_valid_xx = valid_xx.drop(columns = "X_04")

from sklearn import metrics
def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score

xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(new_train_xx,train_yy)
preds = xgb.predict(new_valid_xx)
score1 = lg_nrmse(np.array(valid_yy), preds)

cat = MultiOutputRegressor(CatBoostRegressor(depth = 4, random_state = 42, loss_function = 'RMSE', n_estimators = 3000, learning_rate = 0.03, verbose = 0) 
 ).fit(new_train_xx,train_yy)
preds = cat.predict(new_valid_xx)
score2 = lg_nrmse(np.array(valid_yy), preds)

print(score1, score2)

# 제출해보기...