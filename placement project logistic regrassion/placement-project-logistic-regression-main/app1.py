import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:\placement project logistic regrassion\placement-project-logistic-regression-main\placement.csv')
# print(df.head())
# print(df.shape)
# df = df.iloc[:,1:]
# print(df.head())
# print(df.shape)
# plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
# plt.show()
x = df.iloc[:,0:2]
y = df.iloc[:,3]
# print(x)
# print(x.shape)
# print(y)
# print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.1)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# y_train = scaler.transform(y_train)
# y_train = scaler.transform(x_test)
# print(x_train)
# print(x_test)
clf = LogisticRegression()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_train,y_train.values,clf = clf, legend = 2)
