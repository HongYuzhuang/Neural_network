import pywt
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm

data=pandas.read_csv("c:/users/user/desktop/qbuss6840.csv")
k=[]
n=2
for i in data:
    k.append(data[i])
x1=k[0][1:137]
w1=pywt.Wavelet('sym3')
l= pywt.wavedec(x1,w1,mode='per',level=n)
'''y=pywt.waverec(l, w1,mode='per')
'''
'''
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(np.diff(l[2]),lags=20,ax=ax1)
plt.show()
'''

dl=[]
pl=[]
for i in l:
    dl.append(i[1:]-i[:-1])
for i in dl:
    print(i)
    am=sm.tsa.ARIMA(i,(1,0,0)).fit()
    pl.append(am.predict(start=len(i)+1,end=len(i)+60))
    
for i in range(len(pl)):
    pl[i][0]+=l[i][-1]
    if i==0:
        a=int((40/(2**(n))))+1
    else:
        if i==1:
            a=a
        else:
            a=2*a
    for j in range(a):
        pl[i][j+1]+=pl[i][j]
    pl[i]=pl[i][:a+1]
print(len(l[0]))
g=zip(pl,l)
m=[]
for (i,j) in g:
    m.append(list(j)+list(i))

print(len(l[0]))
py=pywt.waverec(m, w1,mode='per')
plt.plot(py)
y=k[0][137:]
a=[]

for i in y:
    a.append(i)
plt.plot(k[0])
plt.show()

