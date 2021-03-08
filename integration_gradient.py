# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:10:01 2021

@author: Théo
"""


import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import scipy.fftpack
from math import *


def height_from_grad(p,q,lamb,mu):
    max_pq=4
    P1=np.zeros(p.shape,dtype=complex)
    P2=np.zeros(p.shape,dtype=complex)
    Q1=np.zeros(q.shape,dtype=complex)
    Q2=np.zeros(q.shape,dtype=complex)
    H1=np.zeros(p.shape,dtype=complex)
    H2=np.zeros(p.shape,dtype=complex)
    Z=np.zeros(p.shape)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if abs(p[i,j])<max_pq and abs(q[i,j])<max_pq:
                P1[i,j]=p[i,j]
                Q1[i,j]=q[i,j]
            else:
                P2[i,j]=p[i,j]
                Q2[i,j]=q[i,j]

    P1=np.fft.fft2(P1)
    P2=np.fft.fft2(P2)
    Q1=np.fft.fft2(Q1)
    Q2=np.fft.fft2(Q2)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if i!=0 and j!=0:
                L=(1+lamb)*(i**2 + j**2) + mu*(i**2+j**2)**2
                D1=i*P2[i,j]+j*Q2[i,j]
                D2=-i*P1[i,j]-j*Q1[i,j]
                
                H1[i,j]=D1/L
                H2[i,j]=D2/L
            else:
                H1[0,0]=10 # = average height to confirm
    H1=np.fft.ifft2(H1)
    H2=np.fft.ifft2(H2)     
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):    
            Z[i,j]=H2[i,j]
    return Z      



#### DCT #####
            
def DCT_Poisson(p,q):
    #  An implementation of the use of DCT for solving the Poisson equation,
    #  (integration with Neumann boundary condition) 
    #  Code is based on the description in [1], Sec. 3.4
    
    #  [1] Normal Integration: a Survey - Queau et al., 2017
    
    #  Usage : 
    #  u=DCT_Poisson(p,q) 
    #  where p and q are MxN matrices, solves in the least square sense 
    #  \nabla u = [p,q] , assuming natural Neumann boundary condition
    
    #  \nabla u \cdot \eta = [p,q] \cdot \eta on boundaries
    
    #  Axis : O->y
    #         |
    #         x
    
    #  Fast solution is provided by Discrete Cosine Transform
    
    
    
    # Divergence of (p,q) using central differences

    px = 0.5*(np.concatenate((p[1:,:],[p[-1,:]]),axis=0)-np.concatenate(([p[0,:]],p[:-1,:]),axis=0))
    qy = 0.5*(np.concatenate((q[:,1:],np.array(q[:,-1])[..., None]),axis=1)-np.concatenate((np.array(q[:,0])[..., None],q[:,:-1]),axis=1))  
    # Div(p,q) 
    f = px+qy
    
    # Right hand side of the boundary condition
    b = np.zeros(p.shape)  
    b[0,1:-1] = -p[0,1:-1]
    b[-1,1:-1] = p[-1,1:-1]
    b[1:-1,0] = -q[1:-1,0]
    b[1:-1,-1] = q[1:-1,-1]
    b[0,0] = (1/sqrt(2))*(-p[0,0]-q[0,0])
    b[0,-1] = (1/sqrt(2))*(-p[0,-1]+q[0,-1])
    b[-1,-1] = (1/sqrt(2))*(p[-1,-1]+q[-1,-1])
    b[-1,0] = (1/sqrt(2))*(p[-1,0]-q[-1,0])
    
    # Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0,1:-1] = f[0,1:-1]-b[0,1:-1]
    f[-1,1:-1] = f[-1,1:-1]-b[-1,1:-1]
    f[1:-1,0] = f[1:-1,0]-b[1:-1,0]
    f[1:-1,-1] = f[1:-1,-1]-b[1:-1,-1]
    
    # Modification near the corners (Eq. 54 in [1])
    f[0,-1] = f[0,-1]-sqrt(2)*b[0,-1]
    f[-1,-1] = f[-1,-1]-sqrt(2)*b[-1,-1]
    f[-1,0] = f[-1,0]-sqrt(2)*b[-1,0] 
    f[0,0]  = f[0,0]-sqrt(2)*b[0,0] 
    # Cosine transform of f
    fcos=scipy.fftpack.dct(f,type=2)
    print("fcos:",fcos[40:50,40:50])
    
    # Cosine transform of z 
    # x=np.zeros(p.shape)
    # y=np.zeros(p.shape)
    z_bar_bar=np.zeros(p.shape)
    for i in range(p.shape[0]):
       for j in range(p.shape[1]): 
           # x[i,j]=j
           # y[i,j]=i
           denom = 4*((sin(0.5*pi*j/p.shape[1]))**2 + (sin(0.5*pi*i/p.shape[0]))**2)
           z_bar_bar[i,j] = -fcos[i,j]/max(10e-16,denom)
    # ind=np.where(denom<10e-16)
    # z_bar_bar = -fcos/max(10e-16,denom)
    print(z_bar_bar[50:55,50:55])
    
    
    
    # Inverse cosine transform :
    z = scipy.fftpack.idct(z_bar_bar)
    z=z-np.min(z) # Z known up to a positive constant, so offset it to get from 0 to max
    
    return z



###################################




from scipy.io import loadmat

vase=loadmat('D:/Documents/vase.mat')
p=vase['p']
p=p[82:311,89:181]
q=vase['q']
q=q[82:311,89:181]
u=vase['u']
u=u[82:311,89:181]
#mask=vase['mask']
#mask=mask[82:311,89:181]
#std_noise = 0.005*np.max(np.sqrt(p**2+q**2))
#p=p*mask
#q=q*mask
#p=p+std_noise*np.random.normal(size=p.shape)
#q=q+std_noise*np.random.normal(size=q.shape)
#p=p*mask
#q=q*mask

#z_vase=height_from_grad(p, q, lamb=0.1, mu=10)
z_vase=DCT_Poisson(p, q)
#print(z_vase[150:180,150:180])

# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# X,Y = np.meshgrid(range(p.shape[1]),range(p.shape[0]))
# print(X.shape)
# print(Y.shape)
# print(z_vase.shape)
# ax.plot_surface(X, Y, z_vase, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# #ax.plot_wireframe(X, Y, z_vase, color='black')
# ax.set_title('surface')
# fig1= plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot_surface(X, Y, u, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')


