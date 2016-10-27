import pywt
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm

data=pandas.read_csv("c:/users/user/desktop/qbuss6840.csv")
k=[]
n=3
for i in data:
    k.append(data[i])
x1=k[0][1:137]
w1=pywt.Wavelet('db2')
l= pywt.wavedec(x1,w1,mode='per',level=n)
l[2][:]=0
l[3][:]=0
pl=[]
pl.append(sm.tsa.ARIMA(l[0],(1,0,0)).fit().predict(start=len(l[0]),end=len(l[0])+5))
pl.append(sm.tsa.ARIMA(l[1],(1,0,0)).fit().predict(start=len(l[1]),end=len(l[1])+5))
print(len(pl[0]))

a=[0]
pl.append(a*(len(pl[0])*2))
pl.append(a*(len(pl[0])*4))
py=pywt.waverec(pl, w1, mode='per')
plt.plot(list(k[0][137:]))
plt.plot(py)
plt.show()