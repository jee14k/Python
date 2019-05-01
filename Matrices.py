
# coding: utf-8

# In[1]:

u=2.5+3j
v=2
w=u+v


# In[2]:

from math import sin
r=sin(w)


# In[20]:

from cmath import sin,sinh
from numpy.lib.scimath import *
from cmath import *
from math import sqrt
from cmath import sqrt
import numpy
r=sin(w)
r


# In[11]:

r1=sin(8j)
r1


# In[12]:

r2=1j*sinh(8)
r2


# In[15]:

q=8
exp(1j*q)


# In[16]:

cos(q)+1j*sin(q)


# In[19]:

sqrt(-1)


# In[21]:

sqrt(-1)


# 

# In[8]:

import numpy as np


# In[11]:

x=np.array([[1,2],[4,5]])
y=np.array([[1,3],[5,6]])
print(np.add(x,y))


# In[12]:

print(np.subtract(x,y))


# In[13]:

print(np.divide(x,y))


# In[14]:

print(np.multiply(x,y))


# In[15]:

print(np.dot(x,y))


# In[1]:

import numpy as np
matrix = np.matrix([[1,4],[2,0]])
det=np.linalg.det(matrix)
print(det)     


# In[2]:

A=([[1,5,6,7],[8,9,1,0],[2,3,4,5],[4,5,2,3]])
A_det=np.linalg.det(A)
print(A)
print(A_det)


# In[3]:

inverse=np.linalg.inv(matrix)
A_inv=np.linalg.inv(A)
print(inverse)
print(A_inv)
B=np.linalg.inv(A_inv)
print(B)


# In[4]:

A=([[1,3,2],[2,3,1],[4,2,1]])
B=([[2,4,6],[3,2,1],[7,6,2]])
det_A=np.linalg.det(A)
det_B=np.linalg.det(B)
print(det_A)
print(det_B)
inv_A=np.linalg.inv(A)
inv_B=np.linalg.inv(B)
print(inv_A)
print(inv_B)
AB=np.dot(A,B)
AB_INV=np.linalg.inv(AB)
B_INV_A_INV=np.dot(inv_B,inv_A)
print(B_INV_A_INV)
print(AB_INV)


# In[5]:

A=([[1,3],[4,9]])
B=([[4,5],[6,8]])
C=([[6,7],[8,1]])
A_B=np.dot(A,B)
B_C=np.dot(B,C)
A_BC=np.dot(A,B_C)
AB_C=np.dot(A_B,C)
print(A_BC)
print(AB_C)


# In[6]:

E=np.matrix([[1,4,7],[9,1,9],[0,0,1]])
eigvals=np.linalg.eigvals(E)
print(eigvals)


# In[7]:

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
tri_upper_diag=np.triu(a,k=0)
print(tri_upper_diag)
tri_upper_diag_no_diag=np.triu(a,k=1)
print(tri_upper_diag_no_diag)
tri_upper_diag_no_diag=np.triu(a,k=2)
print(tri_upper_diag_no_diag)
tri_upper_diag_no_diag=np.triu(a,k=3)
print(tri_upper_diag_no_diag)


# In[8]:

A=np.array([[3,1],[1,2]])
B=np.array([9,8])
X=np.linalg.solve(A,B)
print(X)


# In[ ]:



