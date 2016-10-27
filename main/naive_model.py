import pandas
import matplotlib.pyplot as plt
import numpy as np
data=pandas.read_csv("c:/users/user/desktop/qbuss6840.csv")
y=data['y']
x144=y[143]
py=[x144]*36
ry=list(y[144:])
plt.plot(ry)
plt.plot(py)
plt.show()
ry=np.mat(ry)
py=np.mat(py)
dy=ry-py
d2y=np.multiply(dy,dy)
RMSE=(np.sum(d2y))**0.5
i=RMSE
print("RMSE = %i " %i)