# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:44:52 2021

@author: Théo
"""

from osgeo import gdal
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import scipy.fftpack
from math import *

#import numpy.fft as fft
#from scipy import fftpack

def Kalman_DTM(z,u,dx,dy):
    F=np.array([[1/3,-dx/2,-dy/2],[0,1/3,0],[0,0,1/3]]) # transition matrix
    Fx=np.array([[1/3,-dx/2,0],[0,1/3,0],[0,0,1/3]])
    Fy=F=np.array([[1/3,0,-dy/2],[0,1/3,0],[0,0,1/3]])
    
    L=10 #nombre grand car P00 inconnu to be set
    P=np.zeros(z.shape[0],z.shape[1],3,3)
    P[0,:]=np.array([[L,0,0],[0,L,0],[0,0,L]])
    P[:,0]=np.array([[L,0,0],[0,L,0],[0,0,L]])
    
    B=np.array([[dx,dy],[0,0],[0,0]])
    
    x=np.zeros(z.shape[0],z.shape[1],3)
    x[0,:,0]=z[0,:]
    x[:,0,0]=z[:,0]
    # x[0,:,1]=model_bias[0,:,0]
    # x[:,0,1]=model_bias[:,0,0]
    # x[0,:,2]=model_bias[0,:,1]
    # x[:,0,2]=model_bias[:,0,1]
    Q=np.array([[0.001,0,0],[0,0.003,0],[0,0,0.003]]) # to be set
    R=0.03
    
    for i in range(1,z.shape[0]):
        for j in range(1,z.shape[1]):
            # step 1
            x[i,j]=F@x[i-1,j-1] + Fx@x[i,j-1] + Fy@x[i-1,j]
            x[i,j]+=B@u[i,j]
            
            # step 2
            P[i,j]=F@P[i-1,j-1]@F.T + Fx@P[i,j-1]@Fx.T + Fy@P[i-1,j]@Fy.T + Q*dx
            
            # step 3
            y=z[i,j]-x[i,j,0]
            
            # step 4
            S=P[i,j,0,0]+R
            
            # step 5
            K=P[i,j,:,0]/S
            
            # step 6
            x[i,j]+=y*K
            
            # step 7
            A=np.column_stack((K,np.zeros(3)))
            A=np.column_stack((A,np.zeros(3)))
            P[i,j]=(np.eye(3)-A)@P[i,j]
        
        
    z_kalman=x[:,:,0]
    return z_kalman       
            
            


import D:\Documents\integ_normale as integ
from mpl_toolkits.mplot3d import Axes3D 

strImgFile = './Data/SRTM30m/geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
#strImgFile = './Data/LiDAR/geo10Md3psi_v-psiN-Nrg-Naz-NazEH.tif'
gdal.UseExceptions()
ds = gdal.Open(strImgFile) # Data Stack

strImgFile_z = './Data/SRTM30m/geo10Md2zSRTM.tif'
#strImgFile_z = './Data/LiDAR/geo10mLiGLTd3LiDTM-CHM-AGB.tif'
gdal.UseExceptions()
ds_z = gdal.Open(strImgFile_z) # Data Stack

I=10 #nb de découpage sur les x
J=10 # nb de découpage sur les y

xp = 20
yp = 20
#########################           SRTM
ox_srtm = 342
oy_srtm = 759
ssImg_omg=np.array(ds.GetRasterBand(5).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_gamm=np.array(ds.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_phi=np.array(ds.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_z=np.array(ds_z.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
                    
                    
n1, n2, n3 = integ.get_normal(ssImg_omg, ssImg_gamm, ssImg_phi)
z = integ.get_z(n1, n2, n3, ssImg_z[0,0], xp, yp)

xx, yy = np.meshgrid(range(xp), range(yp))
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
plt.title("Test")
plt.show()

plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, ssImg_z)
plt.title("Check with z_SRTM")
plt.show()             
        
            
            
            
            
            
            
            
            
            