
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('Crop_recommendation.csv')
df.head()
df.info()
df.describe()
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(df, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


train_set.head()
test_set.head()

df.hist(bins=50,figsize=(20,15))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


my_pipeline=Pipeline([
#     ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])


train_set_label=train_set["label"].copy()
train_set.drop('label', inplace=True, axis=1)

train_set.head()


train_set_label.head()


test_set_label=test_set["label"].copy()
test_set.drop('label', inplace=True, axis=1)
#train_set=my_pipeline.fit_transform(train_set)
train_set
#df_label.head()

from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression


model=svc()
mod=LogisticRegression()
model.fit(train_set,train_set_label)
mod.fit(train_set,train_set_label)
#test_set=my_pipeline.fit_transform(test_set)
print(model.score(test_set,test_set_label))
mod.score(test_set,test_set_label)
#test_set[2]

model.predict(([[0.25421252, 0.27579609, 0.01636317, 3.55194693, 0.95636671,
       0.64209145, 0.14771821]]))
from joblib import dump,load
dump(model,"Crop.joblib")
feature=([[0.25421252, 0.27579609, 0.01636317, 3.55194693, 0.95636671,
       0.64209145, 0.14771821]])
result=model.predict(feature)
print("Done")
print(result[0])
