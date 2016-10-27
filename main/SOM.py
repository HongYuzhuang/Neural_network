# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.special import gammainc 
from matplotlib import pyplot as plt
import math
import sys
#核SOM
#尚需提升 1.权重生成（最好从与输入同分布取样）
class SOMFactory(object):
    def __init__(self,step=10000,tube=(20,20)):
        self.Wlrate=0.01
        self.Slrate=0.1e-4
        self.Nsigma=6
        self.step=step
        self.M,self.N=tube
        self.dataMat=[]
        self.WMat=[]
        self.KernelSigma=[]
        self.dimention=0
        self.sampleNum=0
        self.mean=[]
        self.std=[]
    def Normalization(self,datamat):
        m,n=np.shape(datamat)
        for i in range(m):
            U=np.mean(datamat[i,:])
            O=np.std(datamat[i,:])+1.0e-10
            datamat[i,:]=(datamat[i,:]-U)/(O)
            self.mean.append(U)
            self.std.append(O)
        return datamat
    def retriveNorm(self,datamat):
        m,n=np.shape(datamat)
        for i in range(n):
            datamat[:,i]=(np.dot(datamat[:,i],self.std)+self.mean)
        self.mean=[]
        self.std=[]
        return datamat
    def loaddata(self,path):
        data=pd.read_csv(path)
        self.dataMat=np.matrix(data).T
        self.dimention,self.sampleNum=np.shape(self.dataMat)
        self.dataMat=self.Normalization(self.dataMat)
    def distEU(self,a,b):
        #输入两个列向量
        return ((a-b).T*(a-b)).tolist()[0][0]+1.0e-10
    def kernel(self,x,wi,zigmai):
        k=1-gammainc(self.dimention/2,self.distEU(x,wi)/(2*zigmai**2))
       
        return k
    def dis(self,x,y):
        a1=x % self.M
        b1=int(x/self.M)
        a2=y%self.M
        b2=int(y/self.M)
        k=((a1-a2)**2+(b1-b2)**2)/(2*self.Nsigma**2)
        return math.exp(-k)
        
    def train(self):
        self.WMat=np.random.rand(self.dimention,self.M*self.N)
        self.WMat=(self.WMat-0.5)*2
        self.KernelSigma=np.random.rand(1,self.M*self.N)*0.3
        X=self.dataMat
        sigma0=self.Nsigma
        for i1 in range(self.step):
            xi=X[:,np.random.randint(0,self.sampleNum)]
            result=[self.kernel(xi, self.WMat[:,i2], self.KernelSigma[0,i2]) for i2 in range(self.M*self.N)]
            result=np.matrix(result)
            print(result)
            MAindex=result.argmax()
            for i3 in range(self.M*self.N):
                w=np.matrix(self.WMat[:,i3]).T
                o=np.matrix(self.KernelSigma[0,i3]).T
                self.WMat[:,i3]+=(self.Wlrate*self.dis(i3,MAindex)*(xi-w)/o).T.tolist()[0]
                self.KernelSigma[0,i3]+=(self.Slrate*self.dis(i3,MAindex)*(self.distEU(xi,w)/(self.dimention*o**2)-1)/o).T.tolist()[0]
            self.Nsigma=sigma0*math.exp(-2*sigma0*i1/self.step)
    def PF(self,xi):
        result=[self.kernel(xi, self.WMat[:,i2], self.KernelSigma[0,i2]) for i2 in range(self.M*self.N)]
        return (np.matrix(result)).argmax()
    def project(self,datamat=None):
        #输入 m*n维的矩阵，m为维数，n为样本数
        if(datamat==None):
            datamat =self.dataMat
        datamat=self.Normalization(datamat)
        m,n=np.shape(datamat)
        temp=[self.PF(datamat[:,i]) for i in range(n)]
        Xi=[]
        Yi=[]
        f1=open('C:/users/user/desktop/11.csv','w')
        sys.stdout=f1
        tem='{},{},'
        for  i in temp:
            X=i % self.M
            Xi.append(X)
            Y=int(i/self.M)
            Yi.append(Y)
            print(tem.format(X,Y))
        f1.close()
        plt.scatter(Xi,Yi)      
        plt.show()
                
a=SOMFactory()
a.loaddata('c:/users/user/desktop/new1.csv')
a.train()
a.project()