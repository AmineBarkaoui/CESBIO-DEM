from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
import scipy.fftpack
from math import *

#import numpy.fft as fft
#import numpy.linalg as npl
#from scipy import fftpack
#from scipy import signal
#import scipy.signal as ss
#from scipy.signal import convolve
#import csv
#}}}
#

strImgFile = 'D:\Documents\geo10Md3thtT-thtI-psiN-Nrg-Naz-NazEH.tif'
gdal.UseExceptions()
ds = gdal.Open(strImgFile) # Data Stack

ds_band1 = np.array(ds.GetRasterBand(1).ReadAsArray())
ds_band2 = np.array(ds.GetRasterBand(2).ReadAsArray())
ds_band3 = np.array(ds.GetRasterBand(3).ReadAsArray())
ds_band4 = np.array(ds.GetRasterBand(4).ReadAsArray())
ds_band5 = np.array(ds.GetRasterBand(5).ReadAsArray())
ds_band6 = np.array(ds.GetRasterBand(6).ReadAsArray())

ds_band = np.array(ds.GetRasterBand(3).ReadAsArray(300,725,200,200)) #ox,oy,dx,dy
#plt.figure(figsize=(20,30))
plt.imshow( ds_band*30.0)
plt.show()



# =============================================================================

def lissage(Lx,Ly,p):
    '''Fonction qui débruite une courbe par une moyenne glissante
    sur 2P+1 points'''
    Lxout=[]
    Lyout=[]
    Lxout = Lx[p: -p]
    for index in range(p, len(Ly)-p):
        average = np.mean(Ly[index-p : index+p+1])
        Lyout.append(average)
    return Lxout,Lyout



def doublon(Array):
    list_arr=list(Array)
    index=[]
    unique=[]
    for i in range(len(list_arr)):
        if not(list_arr[i] in unique):
            unique.append(list_arr[i])
            index.append(i)
    return unique, index

    
# =============================================================================



def Lisser(x,y,rhog):

    # fenetre de viualisation
    xmin, xmax = min(x)-.02, max(x)+0.02
    
    
    # Q0. Paramètres problème
    ##########################
    # nb de données
    n = len(x)-1
    # liste des h
    h = x[1:]-x[:-1]
    # nb de points pour pour tracé
    neval = 1201 # 1200 segments
    
    # force de lissage
    rhoGlobal = rhog
    # poids de chacune des données par rapport aux autres
    rhoRelatif = np.ones(len(x))
    #exemple : 
    # rhoRelatif[4] = 100000
    # Au bilan
    rho = rhoGlobal*rhoRelatif  
    
    
    # 2. Détermination des 4-uplets de  la spline d'ajustement
    #################################################################
    
    # a. calcul des sigma''
    #==============================
    
    # i. construction systeme lineaire
    alphaj = 6./(rho[2:n-1]*h[1:n-2]*h[2:n-1])
    betaj = h[1:n-1] - 6.*(h[1:n-1]+h[2:n])/(rho[2:n]*(h[1:n-1]**2)*h[2:n]) - 6.*(h[0:n-2]+h[1:n-1])/(rho[1:n-1]*(h[1:n-1]**2)*h[0:n-2])
    gammaj = 2.*(h[0:n-1]+h[1:n])+ 6./(rho[2:n+1]*h[1:n]**2) + 6./(rho[0:n-1]*h[0:n-1]**2) + 6.*((h[0:n-1]+h[1:n])**2)/(rho[1:n]*(h[0:n-1]**2)*h[1:n]**2)
    deltaj = betaj
    epsj = alphaj
    chij =  6.*((y[2:]-y[1:n])/h[1:n]-(y[1:n]-y[:n-1])/h[:n-1])
                    
    # A = np.diag(alphaj,-2)+np.diag(betaj,-1)+np.diag(gammaj)+np.diag(deltaj,1)+np.diag(epsj,2)
    # ou mieux : en mode sparse
    A = sp.diags([alphaj, betaj, gammaj, deltaj, epsj ], [-2,-1,0,+1,+2], format="csc")
    B = chij
    
    # ii. resolution systeme lineaire
    # -> on fait du solveur direct ce coup-ci mais on pourrait recoder Gauss
    sigma_seconde = np.zeros(n+1)
    # sigma_seconde[1:-1] = npl.solve(A,B)
    # ou mieux : en mode sparse
    sigma_seconde[1:-1] = spl.spsolve(A,B)
    
    
    # b. Calcul des sigma'''
    #==============================
    sigma_tierce = np.zeros(n+1)
    sigma_tierce[:-1] = (sigma_seconde[1:]-sigma_seconde[:-1])/h
    
    # c. Calcul des sigma
    #==============================
    sigma = np.zeros(n+1)
    sigma[0] = y[0] - sigma_seconde[1]/(rho[0]*h[0])
    sigma[n] = y[n] - sigma_seconde[n-1]/(rho[n]*h[n-1])
    sigma[1:n] = y[1:n] - (sigma_seconde[2:]-sigma_seconde[1:n])/(rho[1:n]*h[1:n]) + (sigma_seconde[1:n]-sigma_seconde[:n-1]) / (rho[1:n]*h[:n-1])
    
    # d. calcul des sigma'
    #============================
    sigma_prime = np.zeros(n+1)
    sigma_prime[:-1] = (sigma[1:]-sigma[:-1])/h-h/6*(sigma_seconde[1:]+2*sigma_seconde[:-1])
    sigma_prime[-1] = sigma_prime[-2]+h[-1]*sigma_seconde[-2]+(h[-1]**2)/2.*sigma_tierce[-2]
    
    # 3. Évaluation de la spline de lissage
    ##############################################
    
    # a. évaluation directe en une multitude de points :
    def eval_spline(xeval,x,sigma,sigma_prime,sigma_seconde,sigma_tierce) :
        A,B,C,D,j = sigma[0],sigma_prime[0],0.,0.,-1
        Sigmaxx = []
        for xx in xeval :
            while j<n and xx >= x[j+1] :
                j = j+1
                A,B,C,D = sigma[j],sigma_prime[j],sigma_seconde[j],sigma_tierce[j]
            hxx = xx-x[max(0,j)]
            sigmaxx = A+hxx*(B+hxx*(C/2+hxx*D/6))
            Sigmaxx = Sigmaxx + [sigmaxx]
        Sigmaxx = np.array(Sigmaxx)
        return Sigmaxx
    
    # b. évaluation de la spline aux neval points
    x_graphe = np.linspace(xmin,xmax,neval)
    sigma_graphe = eval_spline(x_graphe,x,sigma,sigma_prime,sigma_seconde,sigma_tierce)

    return x_graphe, sigma_graphe



def get_equation(x,y,d):
    degree = d
    coefs, res, _, _, _ = np.polyfit(x,y,degree, full = True)
    ffit = np.poly1d(coefs)
    #print (ffit)
    if res.size==0:
        res=[0]
    conf = sqrt(res[0])/len(x)
    #print(conf)
    return ffit,conf,coefs



def Afficher_interpolation(x,y,equation,r):
    #Tracé des données et de la spline d'interpolation
    # Fenêtre de visualisation
    xmin = min(x)-.02
    xmax = max(x)+0.02
    # nb de points pour pour tracé
    neval = 1201 # 1200 segments
    x_g = np.linspace(xmin,xmax,neval)
    #print(get_equation(x_graphe,sigma_graphe,4)[0])
    #print(get_equation(x_graphe,sigma_graphe,4)[1])
    plt.figure()
    plt.plot(x,y,'ob',label=u"données")
    plt.plot(x_g, equation,'-r',label="spline de lissage")
    plt.xlabel("Azimuthal angle")
    plt.ylabel("Range angle")
    plt.xlim(xmin,xmax)
    plt.title(u"lissage de données")
    plt.grid()
    plt.legend(loc="lower left")
    plt.show()
    

# rhoGlobal=[30]
# for R in rhoGlobal:
def get_local_relation(Img_omg,Img_gamm,rho,deg):
    k=20                        #taille de la sous image
    I=floor(Img_omg.shape[1]/k) #nb de découpage sur les y
    J=floor(Img_omg.shape[0]/k) # nb de découpage sur les x
    local_relation_neg=np.zeros((Img_omg.shape[0],Img_omg.shape[1],deg+1))
    local_relation_pos=np.zeros((Img_omg.shape[0],Img_omg.shape[1],deg+1))
    for i in range(I):
        for j in range(J):
                ox=k*j
                dx=k
                oy=k*i
                dy=k
                
                ssImg_omg=Img_omg[oy:oy+dy, ox:ox+dx]
                ssImg_gamm=Img_gamm[oy:oy+dy, ox:ox+dx]
                
                ssImg_omg=ssImg_omg.flatten()
                ssImg_gamm=ssImg_gamm.flatten()
                
                sorted_index=np.argsort(ssImg_omg)
                ssImg_omg= ssImg_omg[sorted_index]
                ssImg_gamm=ssImg_gamm[sorted_index]            
                
                ssImg_omg,ind = doublon(ssImg_omg)
                ssImg_gamm = ssImg_gamm[ind]
                
                ssImg_omg_pos = []
                ssImg_omg_neg = []
                ssImg_gamm_pos = []
                ssImg_gamm_neg = []
                
                for m in range(len(ssImg_omg)):
                    if ssImg_gamm[m] < 0:
                        ssImg_omg_neg.append(ssImg_omg[m])
                        ssImg_gamm_neg.append(ssImg_gamm[m])
                    else:
                        ssImg_omg_pos.append(ssImg_omg[m])
                        ssImg_gamm_pos.append(ssImg_gamm[m])

    
                x_neg = np.array(ssImg_omg_neg)
                y_neg =np.array( ssImg_gamm_neg)
                
                x_pos = np.array(ssImg_omg_pos)
                y_pos =np.array( ssImg_gamm_pos)
                
                
                ######## Relation on the negative range of gamma ########
                if len(x_neg)!=0:
                    #x_g,y_g=Lisser(x_neg,y_neg,rho)
                    if i==j:
                        Afficher_interpolation(x_neg,y_neg,get_equation(x_neg,y_neg,deg)[0](np.linspace(min(x_neg)-0.02,max(x_neg)+0.02,1201)),rho)
                    local_relation_neg[oy:oy+dy, ox:ox+dx] = get_equation(x_neg,y_neg,deg)[2]
                
                    if get_equation(x_neg,y_neg,deg)[1]>0.01:
                        print("########## \n On rentre dans un multilook \n##########")
                        
                        ####### Découpage en 4 sous images avec une mauvaise relation #######
                        K=2 #nb de découpage sur les y
                        L=2 # nb de découpage sur les x
                    
                        for m in range(K):
                            for l in range(L):
                                    
                                ox=int(k*j + m*k/K)
                                dx=int(k/K)
                                oy=int(k*i + l*k/L)
                                dy=int(k/L)
                                ssImg_omg_2=Img_omg[oy:oy+dy, ox:ox+dx]
                                ssImg_gamm_2=Img_gamm[oy:oy+dy, ox:ox+dx]
                                
                                ssImg_omg_2=ssImg_omg_2.flatten()
                                ssImg_gamm_2=ssImg_gamm_2.flatten()
                                
                                sorted_index=np.argsort(ssImg_omg_2)
                                ssImg_omg_2= ssImg_omg_2[sorted_index]
                                ssImg_gamm_2=ssImg_gamm_2[sorted_index]
                                                        
                                ssImg_omg_2,ind = doublon(ssImg_omg_2)
                                ssImg_gamm_2 = ssImg_gamm_2 [ind]
                                
                                ssImg_omg_neg_2 = []
                                ssImg_gamm_neg_2 = []
                                
                                for n in range(len(ssImg_omg_2)):
                                    if ssImg_gamm_2[n] < 0:
                                        ssImg_omg_neg_2.append(ssImg_omg_2[n])
                                        ssImg_gamm_neg_2.append(ssImg_gamm_2[n])
                                           
                                x_neg_2 = np.array(ssImg_omg_neg_2)
                                y_neg_2 =np.array( ssImg_gamm_neg_2)
    
                                if len(x_neg_2)!=0:                            
                                    #x_g_2,y_g_2=Lisser(x_neg_2,y_neg_2,rho)
                                    #Afficher_interpolation(x_neg_2,y_neg_2,x_g_2, y_g_2,get_equation(x_g_2,y_g_2,deg)[0](x_g_2),rho)
                                    local_relation_neg[oy:oy+dy, ox:ox+dx] = get_equation(x_neg_2,y_neg_2,deg)[2]
                                
                                
                ##### Moyennage plus grand ########    
                # n=0
                # moy=0
                # x_g1=x_g
                # y_g1=y_g
                                   
                # while (get_equation(x_g1,y_g1,4)[1]>0.01) and n<8:
                    
                #     moy+=5
                #     ssImg_omg1=Img_omg[oy-moy:oy+dy+moy, ox-moy:ox+dx+moy]
                #     ssImg_gamm1=Img_gamm[oy-moy:oy+dy+moy, ox-moy:ox+dx+moy]
                    
                #     ssImg_omg1=ssImg_omg1.flatten()
                #     ssImg_gamm1=ssImg_gamm1.flatten()
                    
                #     sorted_index1=np.argsort(ssImg_omg1)
                #     ssImg_omg1= ssImg_omg1[sorted_index1]
                #     ssImg_gamm1=ssImg_gamm1[sorted_index1]
                    
                #     # ssImg_omg= np.array(list(set(ssImg_omg)))
                #     # ssImg_gamm= np.array(list(set(ssImg_gamm)))
                    
                #     ssImg_omg1,ind1 = doublon(ssImg_omg1)
                #     ssImg_gamm1 = ssImg_gamm1[ind1]
                    
                #     ssImg_omg_pos1 = []
                #     ssImg_omg_neg1 = []
                #     ssImg_gamm_pos1 = []
                #     ssImg_gamm_neg1 = []
                    
                #     for k in range(len(ssImg_omg1)):
                #          if ssImg_gamm1[k] < 0:
                #              ssImg_omg_neg1.append(ssImg_omg1[k])
                #              ssImg_gamm_neg1.append(ssImg_gamm1[k])
                #          else:
                #              ssImg_omg_pos1.append(ssImg_omg1[k])
                #              ssImg_gamm_pos1.append(ssImg_gamm1[k])
                
    
    
    
      
    
                #     x_neg1 = np.array(ssImg_omg_neg1)
                #     y_neg1 =np.array( ssImg_gamm_neg1)
                    
                #     x_pos1 = np.array(ssImg_omg_pos1)
                #     y_pos1 =np.array( ssImg_gamm_pos1)
                    
                #     # fenetre de viualisation
                #     xmin = min(x_neg1)-.02
                #     xmax = max(x_neg1)+0.02
                    
                #     x_g1,y_g1=Lisser(x_neg1,y_neg1,R)
                #     Afficher_interpolation(x_neg1,y_neg1,x_g1, y_g1,get_equation(x_g1,y_g1,4)[0](x_g1),R)
                    
                #     n+=1
                #     print(n)
                
                ######## Relation on the positive range of gamma ########
                if len(x_pos)!=0:
                    #x_g,y_g=Lisser(x_pos,y_pos,rho)
                    if i==j:
                        Afficher_interpolation(x_pos,y_pos,get_equation(x_pos,y_pos,deg)[0](np.linspace(min(x_pos)-0.02,max(x_pos)+0.02,1201)),rho)
                    local_relation_pos[oy:oy+dy, ox:ox+dx] = get_equation(x_pos,y_pos,deg)[2]
                    
                    if get_equation(x_pos,y_pos,deg)[1]>0.01:
                        print("########## \n On rentre dans un multilook \n##########")
                        
                        ####### Découpage en 4 sous images avec une mauvaise relation #######
                        K=2 #nb de découpage sur les y
                        L=2 # nb de découpage sur les x
                    
                        for m in range(K):
                            for l in range(L):
                                    
                                ox=int(k*j + m*k/K)
                                dx=int(k/K)
                                oy=int(k*i + l*k/L)
                                dy=int(k/L)
                                ssImg_omg_2=Img_omg[oy:oy+dy, ox:ox+dx]
                                ssImg_gamm_2=Img_gamm[oy:oy+dy, ox:ox+dx]
                                
                                ssImg_omg_2=ssImg_omg_2.flatten()
                                ssImg_gamm_2=ssImg_gamm_2.flatten()
                                
                                sorted_index=np.argsort(ssImg_omg_2)
                                ssImg_omg_2= ssImg_omg_2[sorted_index]
                                ssImg_gamm_2=ssImg_gamm_2[sorted_index]
                                                        
                                ssImg_omg_2,ind = doublon(ssImg_omg_2)
                                ssImg_gamm_2 = ssImg_gamm_2 [ind]
                                
                                ssImg_omg_pos_2 = []
                                ssImg_gamm_pos_2 = []
                                
                                for n in range(len(ssImg_omg_2)):
                                    if ssImg_gamm_2[n] >= 0:
                                        ssImg_omg_pos_2.append(ssImg_omg_2[n])
                                        ssImg_gamm_pos_2.append(ssImg_gamm_2[n])
                                
                                x_pos_2 = np.array(ssImg_omg_pos_2)
                                y_pos_2 =np.array( ssImg_gamm_pos_2)
                                    
                                if len(x_pos_2)!=0:                            
                                    #x_g_2,y_g_2=Lisser(x_pos_2,y_pos_2,rho)
                                    #Afficher_interpolation(x_pos_2,y_pos_2,x_g_2, y_g_2,get_equation(x_g_2,y_g_2,deg)[0](x_g_2),rho)
                                    local_relation_pos[oy:oy+dy, ox:ox+dx] = get_equation(x_pos_2,y_pos_2,deg)[2]
    
    return local_relation_neg,local_relation_pos

Img_omg=np.array(ds.GetRasterBand(5).ReadAsArray(300,725,200,200))
Img_gamm=np.array(ds.GetRasterBand(4).ReadAsArray(300,725,200,200))

relation_neg,relation_pos=get_local_relation(Img_omg, Img_gamm, rho=30, deg=4)

x=Img_omg[40:60,40:60].flatten()
y=Img_gamm[40:60,40:60].flatten()
x_g=np.linspace(min(x)-0.01,max(x)+0.01,1201)
plt.plot(x,y,'o')
plt.plot(x_g,np.poly1d(relation_neg[50,50])(x_g))
plt.plot(x_g,np.poly1d(relation_pos[50,50])(x_g))