import numpy as np
from scipy import optimize

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

def kfm(p, pL):
    return kf(p) if p < pL else kf(pL)

def pminfm(s, pL):
    ps = psf(s)
    f1 = lambda p: -(ps-p)*kfm(p, pL)
    res = optimize.minimize_scalar(f1, bounds=(-1000, ps), method='bounded').x
    return res

def gcmaxfm(s, pL, a, D):
    ps = psf(s)
    p = pminfm(s, pL)
    gc = (ps-p)*kfm(p, pL)/a/D
    return gc

def pfm(gc, s, pL, a, D):
    ps = psf(s)
    pmin = pminfm(s, pL)
    f1 = lambda p: (ps-p)*kfm(p, pL)-a*D*gc
    # if f1(pmin)*f1(ps)>0:
    #     print(s, pL, gc, pmin, ps, gcmaxfm(s, pL, a, D))
    return optimize.brentq(f1, pmin, ps) if pmin < ps else ps

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

def net_gainf(gc, s, pL, a, D, kmax=0.01):
    p = pfm(gc, s, pL, a, D)
    k_cost = kfm(p, pL)/kmax
    A = Af(gc)
    return A*k_cost

# Eller et. al., (2018)
def eller(s, pL, a, D):
    gcmax = gcmaxfm(s, pL, a, D)
    if gcmax > 0:
        f1 = lambda gc: -net_gainf(gc, s, pL, a, D)
        return optimize.minimize_scalar(f1, bounds=(0, gcmax),\
                                        method='bounded').x
    return 0

def simf(s0, t,
         a, D, l, n, Z, kmax=0.01):
    dt = 60*60
    t = int(t*(24*60*60/dt))
    s = np.zeros(t+1)
    s[0] = s0
    net_gain = np.zeros(t)
    gc = np.zeros(t)
    E = np.zeros(t)
    pL = psf(s0)
    PLC = np.zeros(t)
    for i in range(t):
        gc[i] = eller(s[i], pL, a, D)
        net_gain[i] = net_gainf(gc[i], s[i], pL, a, D)*10**6
        E[i] = a*D*gc[i]*dt*l/n/Z
        s[i+1] = max(0, s[i]-E[i])
        PLC[i] = 1-kf(pL)/kmax
        pL = min(pL, pfm(gc[i], s[i], pL, a, D))
    return PLC, s

def simrif(ar, br, cr, ai, bi, ci,
           s0=1, t=100,
           a=1.6, D=0.005, l=1.8*10**-5, n=0.43, Z=0.8, kmax=0.01):
    dt = 60*60
    t = int(t*(24*60*60/dt))
    s = np.zeros(t+1)
    s[0] = s0
    net_gainr, net_gaini = np.zeros(t), np.zeros(t)
    gcr, gci = np.zeros(t), np.zeros(t)
    Er = np.zeros(t)
    pLr, pLi = psf(s0), psf(s0)
    PLCr, PLCi = np.zeros(t), np.zeros(t)
    for i in range(t):
        gcmaxr = gcmaxfm(s[i], pLr, a, D)
        gcmaxi = gcmaxfm(s[i], pLi, a, D)
        gcr_temp = ar*np.exp(-(psf(s[i])/cr)**br)
        gci_temp = ai*np.exp(-(psf(s[i])/ci)**bi)
        gcr[i] = min(gcmaxr*0.99, gcr_temp)
        gci[i] = min(gcmaxi*0.99, gci_temp)
        net_gainr[i] = net_gainf(gcr[i], s[i], pLr, a, D)*10**6
        net_gaini[i] = net_gainf(gci[i], s[i], pLi, a, D)*10**6
        Er[i] = a*D*gcr[i]*dt*l/n/Z
        s[i+1] = max(0, s[i]-Er[i])
        PLCr[i] = 1-kf(pLr)/kmax
        PLCi[i] = 1-kf(pLi)/kmax
        pLr = min(pLr, pfm(gcr[i], s[i], pLr, a, D))
        pLi = min(pLi, pfm(gci[i], s[i], pLi, a, D))
    return sum(net_gaini)

def simESSf(ar, br, cr, ai, bi, ci,
           s0=1, t=100,
           a=1.6, D=0.005, l=1.8*10**-5, n=0.43, Z=0.8, kmax=0.01):
    dt = 60*60
    t = int(t*(24*60*60/dt))
    s = np.zeros(t+1)
    s[0] = s0
    net_gainr, net_gaini = np.zeros(t), np.zeros(t)
    gcr, gci = np.zeros(t), np.zeros(t)
    Er = np.zeros(t)
    pLr, pLi = psf(s0), psf(s0)
    PLCr, PLCi = np.zeros(t), np.zeros(t)
    for i in range(t):
        gcmaxr = gcmaxfm(s[i], pLr, a, D)
        gcmaxi = gcmaxfm(s[i], pLi, a, D)
        gcr_temp = ar*np.exp(-(psf(s[i])/cr)**br)
        gci_temp = ai*np.exp(-(psf(s[i])/ci)**bi)
        gcr[i] = min(gcmaxr*0.99, gcr_temp)
        gci[i] = min(gcmaxi*0.99, gci_temp)
        net_gainr[i] = net_gainf(gcr[i], s[i], pLr, a, D)*10**6
        net_gaini[i] = net_gainf(gci[i], s[i], pLi, a, D)*10**6
        Er[i] = a*D*gcr[i]*dt*l/n/Z
        s[i+1] = max(0, s[i]-Er[i])
        PLCr[i] = 1-kf(pLr)/kmax
        PLCi[i] = 1-kf(pLi)/kmax
        pLr = min(pLr, pfm(gcr[i], s[i], pLr, a, D))
        pLi = min(pLi, pfm(gci[i], s[i], pLi, a, D))
    return PLCr, s
