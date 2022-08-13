import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(99) # Seed 고정

filepath = 'D:/study_home/_data/dacon_antena/'

train = pd.read_csv(filepath + 'train.csv')

train_x = train.filter(regex='X') # Input : X Featrue
train_y = train.filter(regex='Y') # Output : Y Feature

# xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, train_y)
LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)

test = pd.read_csv(filepath + 'test.csv').drop(columns=['ID'])
preds = LR.predict(test)
print(LR.score(train_x, train_y))

submit = pd.read_csv(filepath +'sample_submission.csv')
for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]

submit.to_csv(filepath + 'submission.csv', index=False)


# 0.2842927683724148