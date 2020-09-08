
#data=pd.read_csv('https://github.com/silpasath/churn_dataset/raw/master/CustomerChurn.csv')

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

data=pd.read_csv('https://github.com/silpasath/churn_dataset/raw/master/CustomerChurn.csv')

# Encode as float
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0).astype('float64')



# Encode catigorical variables with 2 levels
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
    data[col] = enc.fit_transform(data[col])
    
# Remove customer ID column
del data["customerID"]

y = np.array(data.Churn.tolist())
x = data.drop('Churn', 1)
x = np.array(data.values)


# Create test and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .2, random_state= 1)

"""
#Threshold for removing correlated variables
threshold = 0.9
# Absolute value correlation matrix
corr_matrix = data.corr().abs()
corr_matrix.head()
# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d columns to remove :' % (len(to_drop)))
print(to_drop)
data = data.drop(columns = to_drop)
to_drop
"""



#CatBoost

from catboost import CatBoostClassifier
# Import functions which will be used in all models
from sklearn.model_selection import cross_val_score, ShuffleSplit
# Create a split object – it will be used in all models.
split = ShuffleSplit(n_splits=3, test_size=0.25, random_state=1111)
# Define an empty list to store precision for different hyperparameters
cbr_precisions = list()

"""
# Values of l2_leaf_reg to be checked
l2_values = [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200]

# Loop over l2_leaf_reg values to find an optimal
for l2 in l2_values: 
    # Create CatBoostClassifier object
    cbr_foropt = CatBoostClassifier(l2_leaf_reg = l2, silent = True)   
    # Fit it to the cross validator
    cv_results_list_cbr = cross_val_score(cbr_foropt, x, y, cv = split, scoring = 'precision')  
    # Compute the mean precision
    cbr_avg_precision = np.mean(cv_results_list_cbr)    
    # Append this mean to the list of precisions
    cbr_precisions.append(cbr_avg_precision)
    
optimal_l2 = l2_values[cbr_precisions.index(max(cbr_precisions))]
print('Optimal l2-value: ', optimal_l2)
"""

# Create CatBoostClassifier with our optimal hyperparameter
cbr = CatBoostClassifier(l2_leaf_reg = 23, silent = True,od_type='Iter', leaf_estimation_method ='Newton', learning_rate=0.057, depth=6, iterations = 877, loss_function='Logloss')
cbr.fit(x_train,y_train)
y_pred = cbr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


from catboost import Pool
x_df = pd.DataFrame(x)
is_cat = (x_df.dtypes != float)
cat_features_index = np.where(is_cat)[0]
pool = Pool(x, y, cat_features=cat_features_index, feature_names=list(x_df.columns))
model = CatBoostClassifier(l2_leaf_reg = 23, silent = True).fit(pool,early_stopping_rounds=10)
y_pred = model.predict(x)
accuracy = accuracy_score(y, y_pred)
print(accuracy)
# Saving model to disk
import pickle
pickle.dump(cbr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(np.array([[1,0,1,0,1,0,1,29.85,29.85,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0]])))

print(cbr.predict(np.array(data.iloc[1,:])))
