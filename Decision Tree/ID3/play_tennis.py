import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#Load dataset
ds = pd.read_csv('PlayTennis.csv')
X = ds.iloc[:,:-1]
y = ds.iloc[:, -1]

for col in X.columns:
    if X[col].dtype=='object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

if y.dtype=='object':
    te = LabelEncoder()
    y = te.fit_transform(y)
else:
    te = None

#model
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

#visualise
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=te.classes_ if te else None, filled=True)
plt.show()

#predict
sample = X.iloc[0:1]
y_pred = clf.predict(sample)[0]
if te:
    y_pred = te.inverse_transform([y_pred])[0]
print(y_pred)