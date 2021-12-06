# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:38:44 2021

@author: doguilmak

dataset: https://www.kaggle.com/nasa/kepler-exoplanet-search-results

"""
#%%
# 1. Importing Libraries

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

#%%
# 2. Data Preprocessing

# 2.1. Importing Data
start = time.time()
df = pd.read_csv('cumulative.csv')
print(df.head())

# 2.2. Remove Unnecessary Columns
df.drop(['koi_pdisposition','rowid','kepid','kepoi_name','kepler_name','koi_teq_err1','koi_teq_err2','koi_tce_delivname'], axis = 1, inplace = True)

# 2.3. Creating correlation matrix heat map
plt.figure(figsize = (40,40))
sns.heatmap(df.corr(), annot=True)

# 2.4. Looking for NaN Values and Replacing Mean Values of the Column
print("Number of NaN values: ", df.isnull().sum().sum())
df.fillna(df.mean(), inplace=True)
print("Number of NaN values: ",df.isnull().sum().sum())

# 2.5. Looking for Duplicated Datas
print("{} duplicated rows.".format(df.duplicated().sum()))

# 2.6. Looking For Anomalies
print(df.describe().T)

# 2.7. Determination of Dependent and Independent Variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["koi_disposition"] = le.fit_transform(df['koi_disposition'])
y = df["koi_disposition"]
X = df.drop("koi_disposition", axis = 1)

# 2.8. Train - Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 2.9. Scaling Datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test)

#%%
# 3 XGBoost

# 3.1 Fit and Train
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 3.2 Creating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test)
print("\nConfusion Matrix(XGBoost):\n", cm2)

# 3.3. Accuracy of XGBoost
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")

# 3.4. Plotting XGBoost Classifier Tree
from xgboost import plot_tree
import matplotlib.pyplot as plt

plot_tree(classifier)
plt.gcf().set_size_inches(200, 70)
plt.show()

# 3.5. Prediction
predict_model_XGBoost = np.array([1, 0, 0, 0, 0, 9.48803557, 2.78E-05, -2.78E-05, 170.53875, 2.16E-03, -2.16E-03,	0.146, 0.318, -0.146, 2.9575, 0.0819, -0.0819, 6.16E+02, 1.95E+01, -1.95E+01, 2.26, 2.60E-01, -1.50E-01, 793, 93.59, 29.45,	-16.65,	35.8, 1, 5455, 81,-81, 4.467, 0.064, -0.096, 0.927,	0.105, -0.061, 291.93423, 48.141651, 15.347]).reshape(1, 41)  # 1st row of dataset.
if classifier.predict(predict_model_XGBoost) == 0:
    print('\nModel predicted as CANDIDATE.')
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')
elif classifier.predict(predict_model_XGBoost) == 1:
    print('\nModel predicted as CONFIRMED.')
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')   
else:
    print('\nModel predicted as FALSE POSITIVE.')
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
