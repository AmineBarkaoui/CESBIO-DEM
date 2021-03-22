# -*- coding: utf-8 -*-
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from math import *
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# strImgFile = './Data/SRTM30m/geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
# #strImgFile = './Data/LiDAR/geo10Md3psi_v-psiN-Nrg-Naz-NazEH.tif'
# gdal.UseExceptions()
# ds = gdal.Open(strImgFile) # Data Stack

# strImgFile_z = './Data/SRTM30m/geo10Md2zSRTM.tif'
# #strImgFile_z = './Data/LiDAR/geo10mLiGLTd3LiDTM-CHM-AGB.tif'
# gdal.UseExceptions()
# ds_z = gdal.Open(strImgFile_z) # Data Stack

# I=10 #nb de découpage sur les x
# J=10 # nb de découpage sur les y

# xp = 20
# yp = 20
# #########################           SRTM
# ox_srtm = 342
# oy_srtm = 759
# ssImg_omg=np.array(ds.GetRasterBand(5).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
# ssImg_gamm=np.array(ds.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
# ssImg_phi=np.array(ds.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
# ssImg_z=np.array(ds_z.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))


########################            LiDAR
#ssImg_omg=np.array(ds.GetRasterBand(4).ReadAsArray(362, 779, 20, 20))
#ssImg_gamm=np.array(ds.GetRasterBand(3).ReadAsArray(362, 779, 20, 20))
#ssImg_phi=np.array(ds.GetRasterBand(1).ReadAsArray(362, 779, 20, 20))
#ssImg_z=np.array(ds_z.GetRasterBand(1).ReadAsArray(373, 540, 20, 20))


def get_normal(az,ran,phi):
    n3_kvh = -np.cos(az)
    n2_kvh = -np.sin(ran)
    n1_kvh = 1 - n2_kvh**2 - n3_kvh**2
    n1_kvh[np.where(n1_kvh<0)] = 0
    n1_kvh = np.sqrt(n1_kvh)
    n3 = n3_kvh*np.sin(phi)
    n2 = n2_kvh*np.sin(phi)*cos(radians(8))
    n1 = n1_kvh*cos(radians(8))
    return n1, n2, n3


def integ_grad(n1,n2,n3,point_prev,xp,yp):
    # d = -point.dot(np.array([n1,n2,n3]))
    
    # plot the surface
#    xx, yy = np.meshgrid(range(xp), range(yp))
#    z = (-n1 * xx - n2 * yy) * 1. /n3
#    print(z[5,5])
#    plt3d = plt.figure().gca(projection='3d')
#    plt3d.plot_surface(xx, yy, z)
#    plt.xlabel('x'); plt.ylabel('y')
#    plt.show()
    
    # integrate under the surface
    f = lambda y,x : (-n1*x-n2*y)*1./n3
    i = spi.dblquad(f,0,xp,lambda x:0, lambda x:yp)
    return (i[0]/(xp*yp)) + point_prev[2]

# integ_grad(1,0,1,np.array([5,5,-5]),10,10)

def get_z(n1,n2,n3,xp,yp,init,GCP=None):
    m,n = np.shape(n1)
#    t1 = -n1/n3
#    t2 = -n2/n3
    z = np.zeros(n1.shape)
    z[0,0] = init
#    found = 0
#    pos = 0
#    delta = 0
    for i in range(m):
        for j in range(n): 
            if j != 0:
                point = np.array([(i+0.5)*xp,(j+0.5)*yp,z[i,j-1]])
                z[i,j] = integ_grad(n1[i,j],n2[i,j],n3[i,j],point,xp,yp)
            elif j == 0 and i != 0:
                point = np.array([(i+0.5)*xp,(j+0.5)*yp,z[i-1,j]]) 
                z[i,j] = integ_grad(n1[i,j],n2[i,j],n3[i,j],point,xp,yp)
                
#            if GCP[i,j] != 0:
#                z[i,j] = GCP[i,j]
#                if found == 0:
#                    pos = i*n+j
#                    delta = GCP[i,j] - z[i,j]
#                found = 1
#                
#    condition = False
#    for i in range(m):
#        if condition:
#            break
#        for j in range(n):
#            if i*n+j <= pos:
#                z[i,j] += delta
#            else:
#                condition = True
#                break
        
    return z


# n1, n2, n3 = get_normal(ssImg_omg, ssImg_gamm, ssImg_phi)
# z = get_z(n1, n2, n3, ssImg_z[0,0], xp, yp)

# xx, yy = np.meshgrid(range(xp), range(yp))
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z)
# plt.title("Test")
# plt.show()

# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, ssImg_z)
# plt.title("Check with z_SRTM")
# plt.show()