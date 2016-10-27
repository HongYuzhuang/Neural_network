import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas



file="C:/users/user/desktop/new.csv"
data=pandas.read_csv(file)
x=data.drop('y',axis=1)
y=np.matrix(data['y'])
print(len(y.T))
clf=DecisionTreeRegressor(max_depth=20)
a=clf.fit(x,y.T)
py=clf.predict(x)
x=[i for i in range(len(y.T))] 
plt.plot(x,py)
plt.plot(x,(y.T).tolist())
plt.show()
