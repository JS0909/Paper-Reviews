from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

#1. 데이터
path = 'D:/study_home/_data/dacon_antena/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       ).drop(columns=['ID'])
# print(train_set.shape)  #(39607, 71)      

train_x = train_set.filter(regex='X') # Input : X Featrue : 56
train_y = train_set.filter(regex='Y') # Output : Y Feature : 14

x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,shuffle=True,random_state=1234,train_size=0.8)
# print(train_x.shape,train_y.shape)  #(39607, 56) (39607, 14)     
# print(test_set.shape) # (39608, 56)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
# scaler = StandardScaler() 
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score

# 2. 모델

n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)
# parameters = {'n_estimators':[100], # 디폴트 100/ 1~inf 무한대 
# eta[기본값=0.3, 별칭: learning_rate]
# gamma[기본값=0, 별칭: min_split_loss]
# max_depth[기본값=6]
# min_child_weight[기본값=1] 0~inf
# subsample[기본값=1] 0~1
# colsample_bytree [0,0.1,0.2,0.3,0.5,0.7,1]    [기본값=1] 0~1
# colsample_bylevel': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'colsample_bynode': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'reg_alpha' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=0] 0~inf /L1 절댓값 가중치 규제 
# 'reg_lambda' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=1] 0~inf /L2 절댓값 가중치 규제 
# max_delta_step[기본값=0]

parameters = {'n_estimators':[100],
              'learning_rate':[0.1],
            #   'max_depth': [3],
            #   'gamma' : [1],
            #   'min_child_weight' : [1],
            #   'subsample' : [1],
            #   'colsample_bytree' : [0.5],
            #   'colsample_bylevel': [1],
            #   'colsample_bynode': [1],
            #   'alpha' : [0],
            #   'lambda' : [0]
              } # 디폴트 6 

#2. 모델 
xgb = XGBRegressor(random_state=123,
                    n_estimators=100,
                    tree_method='gpu_hist'
                )

model = GridSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)
import time
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time() - start_time
# model.score(x_test,y_test)
results = model.score(x_test,y_test)

print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('걸린 시간 : ',end_time)
print('model.socre : ',results)
y_summit = model.predict(test_set)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                     )
# print(submission)
for idx, col in enumerate(submission.columns):
    if col=='ID':
        continue
    submission[col] = y_summit[:,idx-1]
print('Done.')
submission.to_csv('test23.csv',index=False)
 
 
 
 
# 최적의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.0634512010221178
# 걸린 시간 :  56.3487868309021
# model.socre :  0.06686437973139016

# sub 06