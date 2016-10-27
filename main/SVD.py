# -*- coding:utf-8 -*-
from numpy import *

eps=1.0e-6
def cosSim(inA,inB):
    denom=linalg.norm(inA)*linalg.norm(inB)
    return float(inA*inB.T)/(denom+eps)
def recommand(dataSet,testVect,r=3,rank=1,distCalc=cosSim):
    m,n=shape(dataSet)
    limit=min(m,n)
    if(r>limit):
        r=limit
    U,S,VT=linalg.svd(dataSet.T)
    V=VT.T
    Ur=U[:,:r]
    Sr=diag(S)[:r,:r]
    Vr=V[:,:r]
    testresult=testVect*Ur*linalg.inv(Sr)
    resultarray=array([distCalc(testresult,vi) for vi in Vr])
    descindx=argsort(-resultarray)[:rank]
    return descindx,resultarray[descindx]
A=mat([[5,5,3,0,5,5],[5,0,4,0,4,4],[0,3,0,5,4,5],[5,4,3,3,5,5]])
new=mat([[5,5,0,0,0,5]])
indx,result=recommand(A, new, r=2, rank=3)
print(indx)
print(result)