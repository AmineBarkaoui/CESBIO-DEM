# -*- coding: utf-8 -*-
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

strImgFile = './Data/geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
gdal.UseExceptions()
ds = gdal.Open(strImgFile) # Data Stack

cols=ds.RasterXSize
rows= ds.RasterYSize
bands=ds.RasterCount
geoT = ds.GetGeoTransform()
srs = ds.GetProjection() # features of geo projection system

dx, dy = ds.RasterXSize, ds.RasterYSize # [XY]resoltion
min_x = min(geoT[0], geoT[0]+dx*geoT[1])
max_x = max(geoT[0], geoT[0]+dx*geoT[1])
min_y = min(geoT[3], geoT[3] + geoT[-1]*dy)
max_y = max(geoT[3], geoT[3] + geoT[-1]*dy) # [ geographic coordinates extent ]

ds_band1 = np.array(ds.GetRasterBand(1).ReadAsArray())
ds_band2 = np.array(ds.GetRasterBand(2).ReadAsArray())
ds_band3 = np.array(ds.GetRasterBand(3).ReadAsArray())
ds_band4 = np.array(ds.GetRasterBand(4).ReadAsArray()) # Pente en range
ds_band5 = np.array(ds.GetRasterBand(5).ReadAsArray()) # Angle azimuthal
ds_band6 = np.array(ds.GetRasterBand(6).ReadAsArray())

def doublon(Array):
    list_arr=list(Array)
    index=[]
    unique=[]
    for i in range(len(list_arr)):
        if not(list_arr[i] in unique):
            unique.append(list_arr[i])
            index.append(i)
    return unique, index

I=10 #nb de découpage sur les x
J=20 # nb de découpage sur les y

for i in range(I):
    for j in range(J):
            if(i==5 and j==5):
                # plt.figure(figsize=(20,30))
                ssImg_omg=np.array(ds.GetRasterBand(5).ReadAsArray(int(500+(200*i)/I) , int(360+(500*j)/J) , int(200/I), int(500/J) )*30.0)
                ssImg_gamm=np.array(ds.GetRasterBand(4).ReadAsArray(int(500+(200*i)/I) , int(360+(500*j)/J) , int(200/I), int(500/J) )*30.0)
                
    
def get_normal(az,ran):
    n3 = -np.cos(az)
    n2 = -np.sin(ran)
    n1 = 1 - n2**2 - n3**2
#    print(n1[np.where(n1<0)])
#    n1[np.where(n1<0)] 
#    n1 = np.sqrt(n1)
    return n1, n2, n3


def integ_grad(n1,n2,n3):
    xp = 10 # Vraie largeur d'un pixel selon x
    yp = 10 # Vraie largeur d'un pixel selon y
    point = np.array([0,0,40])
    d = -point.dot(np.array([n1,n2,n3]))
    
    # plot the surface
    xx, yy = np.meshgrid(range(xp), range(yp))
    z = (-n1 * xx - n2 * yy - d) * 1. /n3
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
    plt.show()
    
    # integrate under the surface
    f = lambda y,x : (-n1*x-n2*y-d)*1./n3
    i = spi.dblquad(f,0,xp,lambda x:0, lambda x:yp)
    
    return (i[0]/(xp*yp)) # Normalisation ?????

integ_grad(1,2,3)


def get_z(n1,n2,n3):
    m,n = np.shape(n1)
#    t1 = -n1/n3
#    t2 = -n2/n3
    z = np.zeros((m,n))
    for i in range(m):
        for j in range(n): 
           if j != 0:
               z[i,j] = z[i,j-1] + integ_grad(n1[i,j],n2[i,j],n3[i,j]) # Attention à se placer au bord de la facette
           elif j == 0 and i != 0:
               z[i,j] = z[i-1,j] + integ_grad(n1[i,j],n2[i,j],n3[i,j])               
    return z


n1, n2, n3 = get_normal(ssImg_omg, ssImg_gamm)
z = get_z(n1, n2, n3)