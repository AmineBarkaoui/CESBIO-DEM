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
            
            



def Kalman1D(zSRTM,u,dx,dy ):  
    # image de taille X*Y
                        
    # d_x, d_y = dérivée partielle
            
    # u: mesure primaire STRM (?)
    
    #xSTRM: vecteur mesure secondaire
                                    
  k=0  
    # vecteur d'état
  xk=[zSRTM[k], 0,0] # z , biais X, biais Y à calculer 
  Pk=np.zeros((3,3))
  
  # matrice de Transition
      # d_x, d_y les dérivées partielles
  
  F_mil=np.array([ 
               [1, -dx,  0 ],
               [0,  1,   0  ],
               [0,  0,   1  ] ])
  
  F_bords=np.array([
               [1, -dx*np.shape(zSRTM)[0], +dy ],
               [0,          1            ,  0  ],
               [0,          0            ,   1 ] ])
  
  #matrice d'observation
  H=np.array([1,0,0]).T
  
  R=np.eye(3)
  
  #mesures en entrée B,u matrice et vecteur des commandes en entrée (forces, terme source)
  # u=np.zeros((len(X)*len(Y),3)) #à remplir 
                                #une matrice de taille X*Y x 3, chaque ligne est l'entrée (mesure, biais x, biais y)
                                # et à chaque itération on prendra u[k]: le kème triplet des entrées (mesure, bx, by) 
  B=np.array([[dx,dy],[0,0],[0,0]])   #à modifier matrice de u
  
  
  C=np.ones(3)  #matrice de zSRTM, mesure secondaire
  #bruit
  z_noise=np.ones( (np.shape(zSRTM)[0]*np.shape(zSRTM)[1],3) )
  
  # bruit et matrice de covariance
  w=np.ones( (np.shape(Mat)[0]*np.shape(Mat)[1],3) )
  Qk=np.eye(3) 
  

  for i in range(np.shape(zSRTM)[0]):
      for j in range(np.shape(zSRTM)[1]):

          # Attribution de la matrice de transition
          if (k%(np.shape(zSRTM)[1])==0):
              F=F_bords
          else:
              F=F_mil
        
          #Prediction
          print('1',F@xk)
          print('2',B*u[k])
          xk_pred=F@xk + B*u[k] + w[k]
          Pk_pred=F@(Pk)@F.T + Qk
          
          
          #Mise à jour
          
          #mesure secondaire y
          yk= C*zSRTM[k] + z_noise[k]
          
          #gain de Kalman
          K=(Pk_pred @H) @np.linalg.inv( ( H@ Pk_pred @H.T ) + R )
          
          #mise à jour de xk et Pk
          xk=xk_pred +K @(yk -H @xk_pred )
          Pk=(np.eye(3) - K @H) @Pk_pred
          
          
          #Iteration suivante
          k+=1
          
          
  return xk,Pk # a modif            
            
            
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

xp = 80
yp = 80
#########################           SRTM
ox_srtm = 350
oy_srtm = 750
ssImg_omg=np.array(ds.GetRasterBand(5).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_gamm=np.array(ds.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_phi=np.array(ds.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_z=np.array(ds_z.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
                    

########################            LiDAR     
ssImg_omg_lidar=np.array(ds2.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
#ssImg_gamm_lidar=np.array(ds2.GetRasterBand(3).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_phi_lidar=np.array(ds2.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_z_lidar=np.array(ds_z2.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))

               
n1, n2, n3 = integ.get_normal(ssImg_omg, ssImg_gamm, ssImg_phi)
z = integ.get_z(n1, n2, n3, ssImg_z[0,0], xp, yp)

relation_neg,relation_pos,omg_range_neg,omg_range_pos=mlk.get_local_relation(ssImg_omg, ssImg_gamm, rho=30, deg=2)
ssImg_gamm_lidar_pred = mlk.predict(ssImg_gamm,ssImg_omg,relation_neg,relation_pos,omg_range_neg,omg_range_pos)

n1_pred, n2_pred, n3_pred = integ.get_normal(ssImg_omg_lidar,ssImg_gamm_lidar_pred,ssImg_phi_lidar)

grad_pred = np.zeros((n1_pred.shape[0],n1_pred.shape[1],2))
grad_pred[:,:,0] = -n1_pred/n3_pred
grad_pred[:,:,1] = -n2_pred/n3_pred

z_kalman = Kalman_DTM(ssImg_z,grad_pred,10,10)



xx, yy = np.meshgrid(range(xp), range(yp))
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, ssImg_z)
plt.title("z_SRTM")
plt.show()             
        
xx, yy = np.meshgrid(range(xp), range(yp))
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z_kalman)
plt.title("z_kalman")
plt.show()
          
xx, yy = np.meshgrid(range(xp), range(yp))
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, ssImg_z_lidar)
plt.title("z_lidar")
plt.show()       