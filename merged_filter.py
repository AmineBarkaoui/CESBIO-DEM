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
#import matplotlib.pyplot as plt
#import scipy.fftpack
#from math import *

import integ_normale as ig
import kalman_filter as kf

# GET SRTM DATA ===============================================================
strImgFile = './Data/SRTM30m/geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
strImgFile_z = './Data/SRTM30m/geo10Md2zSRTM.tif'
gdal.UseExceptions()
ds = gdal.Open(strImgFile)
ds_z = gdal.Open(strImgFile_z)
# =============================================================================

xp = 20
yp = 20
ox_srtm = 402
oy_srtm = 819
ssImg_omg=np.array(ds.GetRasterBand(5).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_gamm=np.array(ds.GetRasterBand(4).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_phi=np.array(ds.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))
ssImg_z=np.array(ds_z.GetRasterBand(1).ReadAsArray(ox_srtm, oy_srtm, xp, yp))

n1, n2, n3 = ig.get_normal(ssImg_omg, ssImg_gamm, ssImg_phi)

def merged_filter():
    
    return

