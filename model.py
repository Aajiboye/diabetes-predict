import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
import requests
import json

df = pd.read_csv('diabetes.csv')
#splitting target (Y) from features (X)
features = df.columns.tolist()
target = features.pop(features.index('Outcome'))

dia_feats = df[features]
dia_target = df[target]
Xtrain, Xtest, ytrain, ytest = train_test_split(dia_feats, dia_target, random_state=5, test_size=0.3)
classifier = LogisticRegression()
classifier.fit(Xtrain, ytrain)
y_pred = classifier.predict(Xtest)
pickle.dump(classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6,148,72, 35, 0, 33.6, 0.627, 50]]))