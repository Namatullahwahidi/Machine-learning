import numpy as np
import pandas as pd 
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


data = pd.read_excel("Iris.xls")

x_data = data.iloc[:,1:4]
y_data = data.iloc[:,-1:]



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x_data)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y_data, test_size = 0.33 , random_state = 0)
scores =[]
methods =[]
#------------------------------Logistic Regression-----------------------------

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_lr_pred = lr.predict(x_test)
print("Logistic regression score:",lr.score(x_test,y_test))
cm0 = confusion_matrix(y_test,y_lr_pred)
print(cm0)
scores.append(lr.score(x_test,y_test))
methods.append("lr")
#------------------------------KNN Classifier----------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski')
knn.fit(x_train,y_train)
y_knn_pred = knn.predict(x_test)
print("KNN score :",knn.score(x_test,y_test))

cm1 =confusion_matrix(y_test,y_knn_pred)
print(cm1)
scores.append(knn.score(x_test,y_test))
methods.append("knn")
# ------------------------------Decision tree classifier-----------------------

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier( criterion = 'gini')
dt.fit(x_train,y_train)
y_dt_pred = dt.predict(x_test)
score = dt.score(x_test,y_test)

print("Decision tree score:",score)

cm2 = confusion_matrix(y_test,y_dt_pred)
print(cm2) 
scores.append(dt.score(x_test,y_test))
methods.append("Dt")

#---------------------------- Random Forest Classifier-------------------------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5, criterion = 'gini')
rfc.fit(x_train,y_train)
y_rfc_pred = rfc.predict(x_test)
cm3 =confusion_matrix(y_test,y_rfc_pred)

print("Random forest score:",rfc.score(x_test,y_test))
print(cm3)
scores.append(rfc.score(x_test,y_test))
methods.append("rfc")
#---------------------------SVM Classification---------------------------------
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)
y_svc_pred = svc.predict(x_test)
print("SVM score:",svc.score(x_test,y_test))
cm4 = confusion_matrix(y_test,y_svc_pred)
print(cm4)
scores.append(svc.score(x_test,y_test))
methods.append("SVM")
#------------------------Naive Bayes Classification----------------------------

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_gnb_pred = gnb.predict(x_test)
print("Gaussian Naive bayes:",gnb.score(x_test,y_test))
cm5 = confusion_matrix(y_test,y_gnb_pred)
print(cm5)
scores.append(gnb.score(x_test,y_test))
methods.append("Gnb")
#------------------------------------------------------------------------------

#Bizim Data----------------------------
'''
x_data=data.iloc[:,1:2].values
y_data=data.iloc[:,2:3].values
z_data=data.iloc[:,3:4].values

plt.figure(figsize =(10,6))
ax = plt.axes(projection='3d')


ax.scatter3D(x_data,y_data,z_data,cmap='binary',color='red')
ax.set_xlabel('Boy',color='blue',size=20)
ax.set_ylabel('Kilo',color='blue',size=20)
ax.set_zlabel('Yas',color='blue',size=20);
'''
#----------------------------------------------


# Visualization of Scores

#s = scores.sort()
plt.figure(figsize =(8,6))
plt.plot(methods,scores)
plt.xlabel("Classification Methods")
plt.ylabel("Accuracy Scores")
plt.show()


