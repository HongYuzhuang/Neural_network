import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn import cross_validation

data=pd.read_csv("C:/users/user/desktop/data/train.csv")
y=data.SalePrice
X=pd.read_csv('C:/users/user/desktop/clean-data.csv')
X=X.drop('SalePrice',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2)

class RBFNN(object):
    def __init__(self,t=0.01):
        self.t=t
        self.Kernel=None
        self.zigma=None
        self.w=None
    def euDis(self,a,b):
        #ÐÐ
        return (a-b)*(a-b).T  
    def transform(self,X):
        X=np.matrix(X)
        Nsample,NDim=np.shape(X)
        phi=np.zeros((Nsample,NDim))
        for i in range(Nsample):
            for j in range(NDim):
                k=self.Kernel[j,:]
                d=self.euDis(X[i,:],k)
                phi[i,j]=np.exp(d/(-2*self.zigma**2))  
        return phi
    def train(self,X_train,y_train):
        X_Ma=np.matrix(X_train)
        (Nsample,NDim)=np.shape(X_Ma)
        y_pred = KMeans(n_clusters=NDim).fit_predict(X_Ma)
        self.Kernel=[np.zeros((1,NDim)) for i in range(NDim)]
        count=[0 for i in range(NDim)]
        for i in range(Nsample):
            count[y_pred[i]]+=1
            self.Kernel[y_pred[i]]+=np.matrix(X_Ma[i,:])
        for i in range(NDim):
            self.Kernel[i]=np.array(self.Kernel[i][0,:]/count[i])
        self.Kernel=np.matrix(self.Kernel)
        a=np.zeros((NDim,NDim))
        
        for i in range(NDim):
            for j in range(NDim):
                a[i,j]=np.sqrt(self.euDis(self.Kernel[i,:],self.Kernel[j,:]))
        d=np.max(a)
                
        self.zigma=float(d)/((2*NDim)**0.5)
        phi=self.transform(X_Ma)
        
        w=np.zeros((NDim,1))
        P=np.eye(NDim)/self.t
        y_M_Train=np.matrix(y_train).T
        for i in range(Nsample):
            k=np.matrix(phi[i,:]).T
            P=P-P*k*k.T*P/(1+k.T*P*k)
            g=P*k
            a=y_M_Train[i,:]-w.T*k
            w=w+g*a
        self.w=w
    def predict(self,X_test):
        phi2=self.transform(X_test)
        return phi2*self.w 
        

rbfnn=RBFNN()
rbfnn.train(X_train,y_train)
pre=rbfnn.predict(X_test)


print("RBF: {0}".format(np.sqrt(metrics.mean_squared_error(y_test,pre))))


