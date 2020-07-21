import numpy as np
from scipy import optimize
import scipy.integrate as integrate

def psf(s, pe=-1.58*10**(-3), b=4.38):
    return pe*s**(-b)

def sf(ps, pe=-1.58*10**(-3), b=4.38):
    return (ps/pe)**(-1/b)

def kf(p, kmax=0.01, p50=-1): # mol m-2 s-1 MPa-1
    slope = 65.15*(-p50)**(-1.25)
    a = -4*slope/100*p50
    return kmax/(1+(p/p50)**a)

def PLCf(p, kmax=0.01):
    return 1-kf(p)/kmax

def pminfm(s, sL, pk, kmax):
    pL = psf(sL)
    ps = psf(s)
    PLCfm = lambda p: PLCf(pL)-(PLCf(pL)-PLCf(p))*pk
    kfm = lambda p: kmax*(1-PLCfm(p))
    f1 = lambda p: -(ps-p)*kfm(p)
    if pk == 1:
        res = optimize.minimize_scalar(f1, bounds=(-100, ps),\
                                       method='bounded').x
        return res
    if pL < ps:
        res = optimize.minimize_scalar(f1, bounds=(pL, ps), method='bounded').x
        return res
    return ps

def gcmaxfm(s, sL, pk, kmax, a, D):
    pL = psf(sL)
    ps = psf(s)
    PLCfm = lambda p: PLCf(pL)-(PLCf(pL)-PLCf(p))*pk
    kfm = lambda p: kmax*(1-PLCfm(p))
    p = pminfm(s, sL, pk, kmax)
    gc = (ps-p)*kfm(p)/a/D
    return gc

def pfm(gc, s, sL, pk, kmax, a, D):
    pL = psf(sL)
    ps = psf(s)
    pmin = pminfm(s, sL, pk, kmax)
    PLCfm = lambda p: PLCf(pL)-(PLCf(pL)-PLCf(p))*pk
    kfm = lambda p: kmax*(1-PLCfm(p))
    f1 = lambda p: (ps-p)*kfm(p)-a*D*gc
    if pmin < ps:
        return optimize.brentq(f1, pmin, ps)
    return ps

def Acif(ci,
         IPAR=0.002, Tc=20, Oa=21000, Vcmax25=0.0001, Tup=50, Tlw=10,\
         alfa=0.1, wPAR=0.15,
         Q10leaf=2, Q10rs=0.57, Q10Kc=2.1, Q10Ko=1.2): # mol m-2 s-1
    
    #Vcmax temperature response (mol m-2 s-1)
    Vcmax = Vcmax25*(Q10leaf**(0.1*(Tc-25)))/((1+np.exp(0.3*(Tc-Tup)))*\
                                              (1+np.exp(0.3*(Tlw-Tc))))
    
    #Photocompensation(Pa)
    photocomp=Oa/(2*2600*(Q10rs**(0.1*(Tc-25))))
    
    #Kc & Ko 
    Kc = 30*(Q10Kc**(0.1*(Tc-25)))
    Ko = 30000*(Q10Ko**(0.1*(Tc-25)))
    
    #A  
    Ac = Vcmax*((ci-photocomp)/(ci+Kc*(1+Oa/Ko)))
    Al = alfa*(1-wPAR)*IPAR*((ci-photocomp)/(ci+2*photocomp))
    Ae = 0.5*Vcmax

    return min(Ac, Al, Ae)

def Agcf(gc, ci, ca):
    return gc*(ca-ci)/101325

def Af(gc, ca=40):
    f1 = lambda ci: Acif(ci)-Agcf(gc, ci, ca)
    ci = optimize.brentq(f1, 0, ca)
    return Acif(ci)

def net_gainf(gc, s, sL, pk, kmax, a, D):
    p = pfm(gc, s, sL, pk, kmax, a, D)
    pL = psf(sL)
    PLCfm = lambda p: PLCf(pL)-(PLCf(pL)-PLCf(p))*pk
    k_cost = 1-PLCfm(p)
    A = Af(gc)
    return A*k_cost

# Subspace ESS
def gcsLf(s, sL, pk, kmax, a, D):
    f1 = lambda gc: -net_gainf(gc, s, sL, pk, kmax, a, D)
    gcmax = gcmaxfm(s, sL, pk, kmax, a, D)
    if gcmax > 0:
        gc = optimize.minimize_scalar(f1, bounds=(0, gcmax),\
                                      method='bounded').x
        return gc
    return 0

# Average B for invader
def avgfI(sLI, sLR, freq, MAP, pk, kmax, a, D, l, n, Z):
    MDP = MAP/365
    gamma = 1/((MDP/freq)/1000)*n*Z
    
    gcsLfI = lambda s: gcsLf(s, sLI, pk, kmax, a, D) if s>sLI else 0
    gcsLfR = lambda s: gcsLf(s, sLR, pk, kmax, a, D) if s>sLR else 0
    Lf = lambda s: a*D*gcsLfR(s)*l/n/Z+s/100
    rLf = lambda s: 1/Lf(s)
    integralf = lambda s: integrate.quad(rLf, s, 1)[0]
    f_no_c = lambda s: 1/Lf(s)*np.exp(-gamma*s-freq*integralf(s))
    f1 = lambda s: net_gainf(gcsLfI(s), s, sLI, pk, kmax, a, D)*f_no_c(s)
    
    res = integrate.quad(f1, sLI, 1)[0]
    return(res)

# ESS derivation
def ESSf(sLR,
         freq, MAP,
         pk, kmax, a, D, l, n, Z):
    f1 = lambda sLI: -avgfI(sLI, sLR, freq, MAP, pk, kmax, a, D, l, n, Z)
    sLI_opt = optimize.minimize_scalar(f1, bounds=(0.15, 0.4),\
                                       method='bounded').x
    return sLI_opt-sLR
