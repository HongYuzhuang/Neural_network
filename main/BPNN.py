import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import math
import sys 
class BPNet(object):
    def logistic(self,net):
        a=0.1
        return 1.0/(1.0+np.exp(-a*net))
    def dlogit(self,net):
        a=0.1
        return a*np.multiply(net,(1.0-net))
    def errofunc(self,inX):
        return math.sqrt(np.sum(inX*inX.T))
    def __init__(self):
        self.eb=0
        self.iterator=0
        self.eta=0.4
        self.mc=0.1
        self.fittedY=[];
        self.maxiter=10000
        self.nHidden=[12];
        self.nLay=1;
        self.nOut=1;
        self.errlist=[]
        self.dataMat=0
        self.classLabels=0
        self.Xmean=[]
        self.Xstd=[]
        self.Xnormalized=False
        self.Ymean=[]
        self.Ystd=[]
        self.Ynormalized=False
        self.hi_w=[]
        self.hi_b=[]
        self.hi_wb=[]
        self.nSampDim=0
    def init_hiddenWB(self,i=0):
        if i==0:
            self.hi_w.append(2.0*(np.random.rand(self.nHidden[i],self.nSampDim)-0.5))
            self.hi_b.append(2.0*(np.random.rand(self.nHidden[i],1)-0.5))
            self.hi_wb.append(np.mat(self.addcol(np.mat(self.hi_w[i]),np.mat(self.hi_b[i]))))
        else:
            self.hi_w.append(2.0*(np.random.rand(self.nHidden[i],self.nHidden[i-1])-0.5))
            self.hi_b.append(2.0*(np.random.rand(self.nHidden[i],1)-0.5))
            self.hi_wb.append(np.mat(self.addcol(np.mat(self.hi_w[i]),np.mat(self.hi_b[i]))))
    def init_Output(self):
        self.out_w=2.0*(np.random.rand(self.nOut,self.nHidden[-1])-1)
        self.out_b=2.0*(np.random.rand(self.nOut,1)-1)
        self.out_wb=np.mat(self.addcol(np.mat(self.out_w),np.mat(self.out_b)))
    def loadDataset(self,filename):
        self.dataMat=[];self.classLabels=[]
        fr=pd.read_csv(filename)
        for columns in fr:
            self.dataMat.append(fr[columns])
        self.classLabels=self.dataMat[0]
        self.dataMat=np.mat(self.dataMat[1:]).T
        m,n=np.shape(self.dataMat)
        self.nSampDim=n;
        self.nSampNum=m;
        self.dataMat=np.append(self.dataMat,np.ones((m,1)),1)
    def normalizeclass(self,dataMat):
        [m,n]=np.shape(dataMat)
        for i in range(n):
            U=np.mean(dataMat[:,i])
            O=np.std(dataMat[:,i])+1.0e-10
            dataMat[:,i]=(dataMat[:,i]-U)/(O)
            self.Ymean.append(U)
            self.Ystd.append(O)
            self.Ynormalized=True
        
        return dataMat
    def normalize(self,dataMat):
        dataMat2=np.zeros_like(dataMat)
        [m,n]=np.shape(dataMat)
        for i in range(n-1):
            U=np.mean(dataMat[:,i])
            O=np.std(dataMat[:,i])+1.0e-10
            dataMat2[:,i]=(dataMat[:,i]-U)/(O)
            self.Xmean.append(U)
            self.Xstd.append(O)
        self.Xnormalized=True
        return dataMat2
    def addcol(self,matrix1,matrix2):
        [m1,n1]=np.shape(matrix1)
        [m2,n2]=np.shape(matrix2)
        if m1!=m2:
            print("different rows,can not merge matrix")
            return;
        mergMat=np.zeros((m1,n1+n2))
        mergMat[:,0:n1]=matrix1[:,0:n1]
        mergMat[:,n1:(n1+n2)]=matrix2[:,0:n2]
        return mergMat
    def predict(self,SampIn):
        '''
        SampIn is a n *m matrix, while n is the predetermined dimention in model fitting
        m is the length of data . 
        SampIN should be instance of matrix
        '''
        SampIn=SampIn.T
        
        for i in range(self.nSampDim):
            SampIn[i,:]=(SampIn[i,:]-self.Xmean[i])/(self.Xstd[i])
        hi_input=self.hi_wb[0]*SampIn 
        hi_output=[]
        for j in range(self.nLay):
            if (j==0):
                hi_output.append(self.logistic(hi_input))
            else :
                hi_output.append(self.logistic(self.hi_wb[j]*self.addcol(hi_output[j-1].T,np.ones((np.shape(SampIn)[1],1))).T))
        hi2out=self.addcol(hi_output[self.nLay-1].T,np.ones((np.shape(SampIn)[1],1))).T
        out_input=self.out_wb*hi2out
        out_output=self.logistic(out_input)
        out_output=out_output*self.Ystd[0]+self.Ymean[0]
        return out_output
    def predictAndAnalyse(self,SampIn,expected):
        '''
        SampIN and expected should have same number of columns, and both of them
        should be instance of matrix
        '''
        out=self.predict(SampIn)
        err=expected-out
        SSE=self.errofunc(err)
        return SSE
    def bpTrain(self,ii=1):
        counter=0
        self.nLay=ii
        SampIn=(self.normalize(self.dataMat)).T
        expected=np.mat(self.classLabels).T
        expected=self.normalizeclass(expected).T
        for j in range(ii):
            self.init_hiddenWB(j)
        self.init_Output()
        dout_wbOld=0.0;
        dhi_wbOld=[];
        for j in range(ii):
            dhi_wbOld.append(0.0);
        for i in range(self.maxiter):
            hi_input=self.hi_wb[0]*SampIn
            hi_output=[]
            for j in range(ii):
                if (j==0):
                    hi_output.append(self.logistic(hi_input))
                else :
                    hi_output.append(self.logistic(self.hi_wb[j]*self.addcol(hi_output[j-1].T,np.ones((self.nSampNum,1))).T))
            hi2out=self.addcol(hi_output[ii-1].T,np.ones((self.nSampNum,1))).T
            out_input=self.out_wb*hi2out
            out_output=self.logistic(out_input)
            err=expected-out_output
            
            sse=self.errofunc(err)
            self.errlist.append(sse)
            if sse<=self.eb:
                self.iterator=i+1
                break;
            DELTA=np.multiply(err,self.dlogit(out_output))
            delta=[]
            for j in range(ii):
                if (j==0):
                    delta.append((np.multiply(self.out_wb[:,:-1].T*DELTA,self.dlogit(hi_output[ii-j-1]))))
                else :
                    temp=self.hi_wb[ii-j]
                    delta.insert(0,np.multiply(temp[:,:-1].T*delta[0],self.dlogit(hi_output[ii-j-1])))
            dout_wb=DELTA*hi2out.T
            dhi_wb=[]
            for j in range(ii):
                if j==0:
                    dhi_wb.append(delta[j]*SampIn.T)
                else:
                    dhi_wb.append(delta[j]*self.addcol(hi_output[j-1].T, np.ones((self.nSampNum,1))))
            if i==0:
                self.out_wb=self.out_wb+self.eta*dout_wb
                for j in range(ii):
                    self.hi_wb[j]+=self.eta*dhi_wb[j]
            else:
                self.out_wb+=(1.0-self.mc)*self.eta*dout_wb+self.mc*dout_wbOld
                counter+=1
                for j in range(ii):
                    self.hi_wb[j]+=(1.0-self.mc)*self.eta*dhi_wb[j]+self.mc*dhi_wbOld[j]
            dout_wbOld=dout_wb;
            for j in range(ii):
                dhi_wbOld[j]=dhi_wb[j]
            self.eta*=0.9
        self.fittedY=out_output
        print(counter)
#
bpnet=BPNet()
bpnet.loadDataset("C:/users/user/desktop/temp.csv")
dataMat1=bpnet.dataMat
classLabels1=bpnet.classLabels
bpnet.nHidden=[10,20,40,40,20,10]
bpnet.bpTrain(6)
print(bpnet.errlist)

X=np.linspace(0,bpnet.maxiter,bpnet.maxiter)
Y=np.log2(bpnet.errlist)
plt.plot(X,Y)
plt.show()
m=bpnet.nSampNum
bpnet.loadDataset('C:/users/user/desktop/temp2.csv')
bpnet.nSampNum=m
py=bpnet.predict(bpnet.dataMat)
py=py.tolist()[0]
print(py)
f1=open('C:/users/user/desktop/temp3.csv','w')
sys.std=f1
for i in py:
    print(i)


