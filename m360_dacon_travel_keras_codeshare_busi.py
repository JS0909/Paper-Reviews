import pandas as pd
import numpy as np 


path = 'D:/study_data/_data/dacon_travel/'
train_data = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_data = pd.read_csv(path + 'test.csv', index_col=0)

train_data['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_data['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_data['Age'].fillna(train_data.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_data['Age'].fillna(test_data.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_data['Age']=np.round(train_data['Age'],0)
test_data['Age']=np.round(test_data['Age'],0)


train_data['NumberOfChildrenVisiting'].fillna(train_data.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_data['NumberOfChildrenVisiting'].fillna(test_data.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)

train_data['DurationOfPitch']=train_data['DurationOfPitch'].fillna(0)
test_data['DurationOfPitch']=test_data['DurationOfPitch'].fillna(0)

train_data['PreferredPropertyStar'].fillna(train_data.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_data['PreferredPropertyStar'].fillna(test_data.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

train_data['NumberOfTrips'].fillna(train_data.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_data['NumberOfTrips'].fillna(test_data.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train_data.loc[ train_data['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test_data.loc[ test_data['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train_data.loc[train_data['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_data.loc[test_data['Gender'] =='Fe Male' , 'Gender'] = 'Female'

alldata = [train_data,test_data]
for dataset in alldata:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = np.array(train_data.columns)
for col in cols:
      if train_data[col].dtype == 'object':
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.fit_transform(test_data[col])
# print(train_data)

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out, [25,50,75])
    iqr =quartile_3-quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
                       
x_data = train_data.drop(['NumberOfFollowups', 'OwnCar', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'ProdTaken'], axis=1)
test_data = test_data.drop(['NumberOfFollowups', 'OwnCar', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting', 'MonthlyIncome'], axis=1)
y_data = train_data['ProdTaken']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.91,shuffle=True,random_state=1234,stratify=y_data)


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=6,shuffle=True,random_state=123)
cat_paramets = {"learning_rate" : [0.01], 'depth' : [8], 'od_pval' : [0.12673190617341812], 
                # 'model_size_reg': [0.54233262345], 
                'fold_permutation_block': [142], 'l2_leaf_reg' :[0.33021257848638497]}

cat = CatBoostClassifier(verbose=0,random_state=96,n_estimators=1304)
model = GridSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1)


from sklearn.metrics import accuracy_score
model.fit(x_train,y_train)   
y_predict = model.predict(x_test)
accscore = accuracy_score(y_test,y_predict)
print('score :', accscore)

model.fit(x_data,y_data) # fitting again for all data
sub_y = model.predict(test_data)
sub_y = np.round(sub_y,0)
submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = sub_y

submission.to_csv(path+'submission.csv',index=False)