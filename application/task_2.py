import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('./parkinsons.data')
df.head()

df.info()

df.describe()

sns.pairplot(data=df[df.columns[0:24]])
plt.show()

all_features=df.loc[:,df.columns!='status'].values[:,1:]
out_come=df.loc[:,'status'].values

out_come

print(out_come[out_come == 1].shape[0], out_come[out_come == 0].shape[0])

scaler=MinMaxScaler((-1,1))
X=scaler.fit_transform(all_features)
y=out_come

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)

print('The accuracy of the XGBoost classifier on training data is : {:.2f}'.format(xgb_clf.score(X_train, y_train)*100))
print('The accuracy of the XGBoost classifier on test data is : {:.2f}'.format(xgb_clf.score(X_test, y_test)*100))

plt.figure(figsize=(12, 8))
plot_importance(xgb_clf)
plt.title("График важности признаков")
plt.show()

y_pred = xgb_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Parkinson's", "Parkinson's"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()