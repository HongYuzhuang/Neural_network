import pywt
x=[3,7,1,1,-2,5,4,6]
(ca,cd)=pywt.dwt(x,'db2')
y=pywt.idwt(ca,cd,'db2')
print(ca)
print(cd)
print(y)