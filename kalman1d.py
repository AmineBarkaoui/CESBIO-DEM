# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:41:37 2021

@author: ztaie
"""
import numpy as np
import random
import matplotlib.pyplot as plt


from pykalman import KalmanFilter
import numpy as np
kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)


from scipy.interpolate import interp1d
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt




# =============================================================================
#               kalman 1d
# =============================================================================


# =============================================================================
# ### Pour du temps
# =============================================================================
def Kalman(Mesures,Estim,Emes,Eestim):
    em=Emes
    ee=Eestim
    estim=Estim
    list_estim=[]
    list_KG=[]
    for i in range(len(Mesures)):
        m=Mesures[i]      
    
        #KG
        KG=(ee)/(em+ee)
        
        #estim
        estim=estim+KG*(m-estim)
        
        #err e
        ee=(1-KG)*ee
        
        list_KG+=[KG]
        list_estim+=[estim]
        
    return list_estim, list_KG

# =============================================================================
# ### Affichage Kalman temp
# =============================================================================

def Afficher_Kalman_1D_temporel():
    N=100
    Incertitude_mes=4
    val_moy=20
    M=np.random.uniform(0,1,N)*Incertitude_mes+val_moy
    
    Estimateur=1
    Incertitude_est= 4
    
    Resu,KG=Kalman(M, Estimateur, Incertitude_mes,Incertitude_est)
    
    
    # plt.axis([0,N,M[0]/1.5,M[0]+Incertitude_mes*5])
    plt.plot(range(N),Resu)
    plt.scatter(range(N),M,c='pink',marker="x",s=1)
    plt.plot(np.ones(N)*(Incertitude_mes/2)+val_moy)
    plt.legend(["resultat","valeur vraie","observations"])
    plt.show()

Afficher_Kalman_1D_temporel()

# =============================================================================
# ### pour de l'espace:
# =============================================================================
#%%

def Kalman_1D_spatiale(Mesures_obs,Valeurs_modele,Estim_init,Err_mes,Err_estim):
    
    #Mesures_obs: valeurs observées
    #Valeurs_modele: valeurs du modele theorique (ex: par integration de la pente)
    
    em=Err_mes
    ee=Err_estim
    estim=Valeurs_modele[0]
    
    list_estim=[]
    list_KG=[]
    
    taille=len(Mesures_obs)
            
        
    for i in range(taille):
        m=Mesures_obs[i]      
    
    
        #KG gain de Kalman
        KG=(Err_estim)/(Err_mes+Err_estim)
        
        
        #estim
        estim=estim+KG*(Mesures_obs[i] - Valeurs_modele[i]) # remplacer estim par valeurs théoriques
        
        
        #err e
        Err_estim=(1-KG)*Err_estim
        
        list_KG+=[KG]
        list_estim+=[estim]
        
    return list_estim, list_KG








def Afficher_Kalman_1D_spatial():
    
    N=80
    val_moy=20
    Incertitude_mes=0.1 #  comment le trouver ?
    Mesures=np.random.uniform(0,1,N)*Incertitude_mes+val_moy 
    x,y=np.linspace(0,N,N),Mesures
    
    
   
    def Lisser(x,y,rhog):
    
        # fenetre de viualisation
        xmin, xmax = min(x)-.5, max(x)+0.5
        
        
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
        
        
        # # 1. Tracer des données
        # #########################
        # plt.figure(0)
        # plt.plot(x,y,'ob')
        
        # plt.xlim(xmin,xmax)
        # plt.ylim(min(y)-1,max(y)+1)
        # plt.title(u"données à lisser")
        # plt.grid()
        # plt.show()
        
        
        
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
        print (ffit)
        return ffit
    def Afficher_interpolation(x_graphe,sigma_graphe,r):
        # 4. Tracé des données et de la spline d'interpolation
        #########################################################
        get_equation(x_graphe,sigma_graphe,2)
        plt.plot(x,y,'ob',label=u"données")
        plt.plot(x_graphe, get_equation(x_graphe,sigma_graphe,3)(x_graphe),'-r',label="spline de lissage")
        # plt.xlim(xmin,xmax)
        # plt.title(u"lissage de données")
        # plt.grid()
        # plt.legend(loc="lower left")
        # plt.show()
        

    # rhoGlobal=[0.1]
    # for R in rhoGlobal:
    #     x_g,y_g=Lisser(x,y,R)
    #     Afficher_interpolation(x_g, y_g,R)
    #     plt.show()



    # plt.figure()
    # plt.scatter(x,y)
    # plt.title('t')
    # plt.show()


    Modele=np.ones(len(Mesures))
    Estimateur=1
    Incertitude_est= 0.1
    
    Resu,KG=Kalman_1D_spatiale(Mesures,Modele,Modele[0],Incertitude_mes,Incertitude_est)
    
    
    # plt.axis([0,N,M[0]/1.5,M[0]+Incertitude_mes*5])
    plt.plot(range(N),Resu)
    plt.scatter(range(N), Mesures ,c='pink',marker="x",s=1)
    plt.plot(np.ones(N)*(Incertitude_mes/2)+val_moy)




# =============================================================================
# ### Affichage Kalman spatial 1D
# =============================================================================
# Afficher_Kalman_1D_spatial()





# =============================================================================
# =============================================================================
# # **************************************************************************
# =============================================================================
# =============================================================================







# =============================================================================
#                           Kalman 2D A COMPLETER
# =============================================================================

############ pb sur u et xSRTM
def Kalman2D(Mat,u,xSRTM,d_x,d_y ):  
    # image de taille X*Y
                        
    # d_x, d_y = dérivée partielle
            
    # u: mesure primaire STRM (?)
    
    #xSTRM: vecteur mesure secondaire
                                    
  k=0  
    # vecteur d'état
  xk=[0, 0 ,0] # z , biais X, biais Y
  Pk=np.zeros((3,3))
  
  # matrice de Transition
      # d_x, d_y les dérivées partielles
  
  F=np.array([ [1, d_x, d_y ],
               [0,  1,   0  ],
               [0,  0,   0  ] ])
  
  #matrice d'observation
  H=np.array([1,0,0]).T
  
  R=np.eye(3)
  
  #mesures en entrée B,u matrice et vecteur des commandes en entrée (forces, terme source)
  # u=np.zeros((len(X)*len(Y),3)) #à remplir 
                                #une matrice de taille X*Y x 3, chaque ligne est l'entrée (mesure, biais x, biais y)
                                # et à chaque itération on prendra u[k]: le kème triplet des entrées (mesure, bx, by) 
  B=np.ones(3)   #à modifier matrice de u
  
  
  C=np.ones(3)  #matrice de xSRTM, mesure secondaire
  z=np.ones( (np.shape(Mat)[0]*np.shape(Mat)[1],3) )
  
  # bruit et matrice de covariance
  w=np.ones( (np.shape(Mat)[0]*np.shape(Mat)[1],3) )
  Qk=np.eye(3) 
  
  
  for i in range(np.shape(Mat)[0]):
      for j in range(np.shape(Mat)[1]):
          #Prediction
          print('1',F@xk)
          print('2',B*u[k])
          xk_pred=F@xk + B*u[k] + w[k]
          Pk_pred=F@(Pk)@F.T + Qk
          
          
          #Mise à jour
          
              #mesure secondaire y
          yk= C*xSRTM[k] + z[k]
              #gain de Kalman
          print('k', H@ Pk_pred @H.T)
          K=(Pk_pred @H) @np.linalg.inv( ( H@ Pk_pred @H.T ) + R )
          
              #mise à jour xk et Pk
          xk=xk_pred +K @(yk -H @xk_pred )
          Pk=(np.eye(3) - K @H) @Pk_pred
          k+=1
          
          
  return xk,Pk # a modif

N=50
dx=0.5
dy=0.5
A=np.ones((N,N))
mesSAR=np.random.uniform(0,1,N)*5+20        # u
mesSRTM=mesSAR+np.random.uniform(20,1,N)    # xSRTM

x,P= Kalman2D(A,mesSAR,mesSRTM,dx,dy)       