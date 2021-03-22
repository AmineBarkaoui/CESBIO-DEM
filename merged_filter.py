# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:21:21 2021

@author: All
"""

from osgeo import gdal
import numpy as np
#import numpy.linalg as npl
#import scipy.sparse as sp
#import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
#import scipy.fftpack
#from math import *

import integ_normale as ig
import kalman_filter as kf
import multilook as mk
     

# DATA ========================================================================
strImgFile = './Data/SRTM30m/geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
strImgFile_z = './Data/SRTM30m/geo10Md2zSRTM.tif'
gdal.UseExceptions()
ds = gdal.Open(strImgFile)
ds_z = gdal.Open(strImgFile_z)

strImgFile = './Data/LiDAR/geo10Md3psi_v-psiN-Nrg-Naz-NazEH.tif'
strImgFile_z = './Data/LiDAR/geo10mLiGLTd3LiDTM-CHM-AGB.tif'
ds_lidar = gdal.Open(strImgFile) 
ds_z_lidar = gdal.Open(strImgFile_z)

geotransform = ds.GetGeoTransform()
geotransform_Lidar = ds_lidar.GetGeoTransform()
oX_srtm = geotransform[0]
oY_srtm = geotransform[3]
oX_lidar = geotransform_Lidar[0]
oY_lidar = geotransform_Lidar[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
xOffset=int((oX_lidar-oX_srtm)/pixelWidth)
yOffset=int((oY_lidar-oY_srtm)/pixelHeight)

geotransform_zLidar = ds_z_lidar.GetGeoTransform()
ox_zlidar = geotransform_zLidar[0]
oy_zlidar = geotransform_zLidar[3]
xOffset_z=int((ox_zlidar-oX_srtm)/pixelWidth)
yOffset_z=int((oy_zlidar-oY_srtm)/pixelHeight)

xp = 80
yp = 80
# =============================================================================

ox_srtm = 350
oy_srtm = 750
ssImg_omg_srtm=np.array(ds.GetRasterBand(5).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_gamm_srtm=np.array(ds.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_phi_srtm=np.array(ds.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_z_srtm=np.array(ds_z.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))


ssImg_omg_lidar=np.array(ds_lidar.GetRasterBand(5).ReadAsArray(ox_srtm-xOffset, oy_srtm-yOffset, xp, yp))
ssImg_gamm_lidar = mk.get_range_prediction(ssImg_omg_srtm,ssImg_gamm_srtm,ssImg_omg_lidar,deg=2)
ssImg_phi_lidar=np.array(ds_lidar.GetRasterBand(1).ReadAsArray(ox_srtm-xOffset, oy_srtm-yOffset, xp, yp))
ssImg_z_lidar=np.array(ds_z_lidar.GetRasterBand(1).ReadAsArray(ox_srtm-xOffset_z, oy_srtm-yOffset_z, xp, yp))

n1, n2, n3 = ig.get_normal(ssImg_omg_srtm, ssImg_gamm_srtm, ssImg_phi_srtm)
z = ig.get_z(n1, n2, n3, ssImg_z_srtm[0,0], xp, yp)

n1_pred, n2_pred, n3_pred = ig.get_normal(ssImg_omg_lidar,ssImg_gamm_lidar,ssImg_phi_lidar)

grad_pred = np.zeros((n1_pred.shape[0],n1_pred.shape[1],2))
grad_pred[:,:,0] = -n1_pred/n3_pred
grad_pred[:,:,1] = -n2_pred/n3_pred

#z_kalman = merged_filter(ssImg_z_srtm,grad_pred,10,10)
z_kalman = kf.Kalman1D(ssImg_z_srtm,grad_pred,10,10)

xx, yy = np.meshgrid(range(xp), range(yp))
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx[1:,1:], yy[1:,1:], z_kalman[1:,1:])
plt.title("z_kalman")
plt.show()
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, ssImg_z_srtm)
plt.title("z_srtm")
plt.show()
ssImg_z_lidar = ssImg_z_lidar - (ssImg_z_lidar[0,0] - ssImg_z_srtm[0,0])
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, ssImg_z_lidar)
plt.title("z_lidar")
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(131)
plt.imshow(z_kalman)
plt.title("z_Kalman")
#plt.colorbar()
plt.subplot(132)
plt.imshow(ssImg_z_srtm)
plt.title("z_srtm")
#plt.colorbar()
plt.subplot(133)
plt.imshow(ssImg_z_lidar)
plt.title("z_lidar")
#plt.colorbar()
plt.show()


print(np.mean(ssImg_z_lidar-ssImg_z_srtm))
