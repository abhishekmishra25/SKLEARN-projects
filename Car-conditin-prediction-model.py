#This is car condition prediction model
''' 
In this i used sklearn for testing,training and predicting
And also used pandas for dataframe 
'''


#module that you need to import
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('car.data')


x = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]
print(x,y)
#converting the data
le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = le.fit_transform(x[:,i])



#converting using maping
label_maping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_maping)
y = np.array(y)


#create model

knn =neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')

x_train ,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

accuracy =  metrics.accuracy_score(y_test,prediction)
print('prediction: ',prediction)
print('accuracy: ',accuracy)

a = 1727
print('actual value: ', y[a])
print('predicted value: ',knn.predict(x)[a])
