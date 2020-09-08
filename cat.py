
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
from datetime import datetime
import lightgbm as lgbm
import warnings
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier


churn_data = pd.read_csv('https://github.com/silpasath/churn_dataset/raw/master/CustomerChurn.csv')

#churn_data['Churn']=np.where(churn_data.Churn =='Yes',1,0)
#churn_data.TotalCharges=churn_data.TotalCharges.replace(' ',np.nan)
churn_data['TotalCharges'] = churn_data['TotalCharges'].replace(" ", np.nan).astype('float64')
#churn_data.dropna(inplace=True)
#churn_data.TotalCharges=churn_data.TotalCharges.astype(float)

"""
#replace 'No internet service' to No for the following columns
replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
for i in replace_cols : 
    churn_data[i]  = churn_data[i].replace({'No internet service' : 'No'})
    
    
def label_encoder(col):
    churn_data[col]=LabelEncoder().fit_transform(churn_data[col])

for cols in churn_data.columns.drop(['customerID','TotalCharges','tenure','MonthlyCharges','Churn']).tolist():
    label_encoder(cols)
"""
enc = LabelEncoder()
encode_columns = ['gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod',
 'Churn']

for col in encode_columns:
    churn_data[col] = enc.fit_transform(churn_data[col])
    
del churn_data["customerID"]
    
X=churn_data.drop(['Churn'],axis=1)
y=churn_data[['Churn']]
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=y)


your_model=CatBoostClassifier(eval_metric='AUC')
your_model.fit(X_train,y_train)
pred=your_model.predict(X_test)
print("Accuracy of {0} : {1}".format(str(your_model)[:],accuracy_score(y_test,pred)))
print("AUC :",roc_auc_score(y_test,pred))
    



