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
    
    L=1000 #nombre grand car P00 inconnu to be set
    P=np.zeros((z.shape[0],z.shape[1],3,3))
    P[0,:]=np.array([[L,0,0],[0,L,0],[0,0,L]])
    P[:,0]=np.array([[L,0,0],[0,L,0],[0,0,L]])
    
    B=np.array([[dx,dy],[0,0],[0,0]])
    
    x=np.zeros((z.shape[0],z.shape[1],3))
    x[0,:,0]=z[0,:]
    x[:,0,0]=z[:,0]
    # x[0,:,1]=model_bias[0,:,0]
    # x[:,0,1]=model_bias[:,0,0]
    # x[0,:,2]=model_bias[0,:,1]
    # x[:,0,2]=model_bias[:,0,1]
    Q=np.array([[5,0,0],[0,0.17,0],[0,0,0.17]]) # to be set
    R=5
    
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
            
            



def Kalman1D(z,u,dx,dy ):  
    # image de taille X*Y
                        
            
    # u: gradient SAR
    
    #xSTRM: vecteur mesure secondaire
                                     
    # vecteur d'état 
    Fx=np.array([[1,-dx,0],[0,1,0],[0,0,1]]) # transition matrix
    Fy=F=np.array([[1,0,-dy],[0,1,0],[0,0,1]])
    
    L=1000 #nombre grand car P00 inconnu to be set
    P=np.zeros((z.shape[0],z.shape[1],3,3))
    P[0,:]=np.array([[L,0,0],[0,L,0],[0,0,L]])
    P[:,0]=np.array([[L,0,0],[0,L,0],[0,0,L]])
    
    B=np.array([[dx,dy],[0,0],[0,0]])
    
    x=np.zeros((z.shape[0],z.shape[1],3))
    x[0,0,0]=z[0,0]
    # x[0,:,1]=model_bias[0,:,0]
    # x[:,0,1]=model_bias[:,0,0]
    # x[0,:,2]=model_bias[0,:,1]
    # x[:,0,2]=model_bias[:,0,1]
    Q=np.array([[5,0,0],[0,0.17,0],[0,0,0.17]]) # to be set
    R=5
    
    for i in range(1,z.shape[0]):
        for j in range(1,z.shape[1]):
            if j==0:
                # step 1
                x[i,j]=Fy@x[i-1,j] 
                x[i,j]+=B@u[i,j]
                
                # step 2
                P[i,j]=Fy@P[i-1,j]@Fy.T + Q*dy
                
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
                
            else:
                # step 1
                x[i,j]=Fx@x[i,j-1] 
                x[i,j]+=B@u[i,j]
                
                # step 2
                P[i,j]=Fx@P[i,j-1]@Fx.T + Q*dx
                
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
  
          
    return x[:,:,0]          
            
            
import integ_normale as integ
import multilook as mlk
from mpl_toolkits.mplot3d import Axes3D 

strImgFile = './Data/SRTM30m/geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
strImgFile2 = './Data/LiDAR/geo10Md3psi_v-psiN-Nrg-Naz-NazEH.tif'
gdal.UseExceptions()
ds = gdal.Open(strImgFile) # Data Stack
ds2 = gdal.Open(strImgFile2)

strImgFile_z = './Data/SRTM30m/geo10Md2zSRTM.tif'
strImgFile_z2 = './Data/LiDAR/geo10mLiGLTd3LiDTM-CHM-AGB.tif'
gdal.UseExceptions()
ds_z = gdal.Open(strImgFile_z) # Data Stack
ds_z2 = gdal.Open(strImgFile_z2)


geotransform = ds.GetGeoTransform()
geotransform_zLidar = ds2.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
originX_Lidar = geotransform_zLidar[0]
originY_Lidar = geotransform_zLidar[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
xOffset=int((originX_Lidar-originX)/pixelWidth)
yOffset=int((originY_Lidar-originY)/pixelHeight)


xp = 80
yp = 80
#########################           SRTM
ox_srtm = 350
oy_srtm = 750
azimuth_SRTM=np.array(ds.GetRasterBand(5).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
range_SRTM=np.array(ds.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
phi_SRTM=np.array(ds.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
z_SRTM=np.array(ds_z.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
                    

########################            LiDAR     
azimuth_lidar=np.array(ds2.GetRasterBand(4).ReadAsArray(ox_srtm-xOffset, oy_srtm-yOffset, xp, yp))
#range_lidar=np.array(ds2.GetRasterBand(3).ReadAsArray(ox_srtm-xOffset, oy_srtm-yOffset, xp, yp))
phi_lidar=np.array(ds2.GetRasterBand(2).ReadAsArray(ox_srtm-xOffset, oy_srtm-yOffset, xp, yp))


geotransform = ds.GetGeoTransform()
geotransform_zLidar = ds_z2.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
originX_Lidar = geotransform_zLidar[0]
originY_Lidar = geotransform_zLidar[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
xOffset=int((originX_Lidar-originX)/pixelWidth)
yOffset=int((originY_Lidar-originY)/pixelHeight)

z_lidar=np.array(ds_z2.GetRasterBand(1).ReadAsArray(ox_srtm-xOffset, oy_srtm-yOffset, xp, yp))

               
n1, n2, n3 = integ.get_normal(azimuth_SRTM, range_SRTM, phi_SRTM)
z = integ.get_z(n1, n2, n3, z_SRTM[0,0], xp, yp)

range_lidar_pred=mlk.get_range_prediction(azimuth_SRTM,range_SRTM,2)
# relation_neg,relation_pos,omg_range_neg,omg_range_pos=mlk.get_local_relation(azimuth_SRTM, range_SRTM, rho=30, deg=2)
# range_lidar_pred = mlk.predict(range_SRTM,azimuth_SRTM,relation_neg,relation_pos,omg_range_neg,omg_range_pos)

n1_pred, n2_pred, n3_pred = integ.get_normal(azimuth_lidar,range_lidar_pred,phi_lidar)

grad_pred = np.zeros((n1_pred.shape[0],n1_pred.shape[1],2))
grad_pred[:,:,0] = -n1_pred/n3_pred
grad_pred[:,:,1] = -n2_pred/n3_pred


pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
z_kalman = Kalman_DTM(z_SRTM,grad_pred,pixelWidth,pixelHeight)


############### Affichage Z SRTM ###############
# xx, yy = np.meshgrid(range(xp), range(yp))
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z_SRTM)
# plt.title("z_SRTM")
# plt.show()   
plt.imshow(z_SRTM)
plt.title("Z SRTM")
plt.show()         
     
############### Affichage Z Kalman ###############   
# xx, yy = np.meshgrid(range(xp), range(yp))
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z_kalman)
# plt.title("z_kalman")
# plt.show()
plt.imshow(z_kalman)
plt.title("Z Kalman")
plt.show()
          
############### Affichage Z LiDAR ###############
# xx, yy = np.meshgrid(range(xp), range(yp))
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z_lidar)
# plt.title("z_lidar")
# plt.show()    
plt.imshow(z_lidar)
plt.title("Z LiDAR")
plt.show()   