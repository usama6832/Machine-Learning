# KNN (K Nearest Neighbors) algorithm

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

iris.feature_names
iris.target_names

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
df['target'] = iris.target
df.head()
df[df.target==1].head()
df[df.target==2].head()
df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


df[45:55]
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

import matplotlib.pyplot as plt
# matplotlib inline
# Sepal length vs Sepal Width (Setosa vs Versicolor)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

# Petal length vs Pepal Width (Setosa vs Versicolor)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

# Train test split

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
len(X_train)
len(X_test)

#Create KNN (K Neighrest Neighbour Classifier)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
KNeighborsClassifier(n_neighbors=10)
knn.score(X_test, y_test)
knn.predict([[4.8,3.0,1.5,0.3]])

# Plot Confusion Matrix

from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

# matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# Print classification report for precesion, recall and f1-score for each classes

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
