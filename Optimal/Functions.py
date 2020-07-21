import numpy as np
from scipy import optimize
import scipy.integrate as integrate

def psf(s, pe=-1.58*10**(-3), b=4.38):
    return pe*s**(-b)

def kf(p, kmax=0.01, p50=-1): # mol m-2 s-1 MPa-1
    slope = 65.15*(-p50)**(-1.25)
    a = -4*slope/100*p50
    return kmax/(1+(p/p50)**a)

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
    f1 = lambda ci:Acif(ci)-Agcf(gc, ci, ca)
    ci = optimize.brentq(f1, 0, ca)
    return Acif(ci)

def Egcf(gc, a=1.6, D=0.005): # mol m-2 leaf s-1
    return a*gc*D

def Epf(p, ps): # mol m-2 leaf s-1
    return (ps-p)*kf((ps+p)/2)

def pminf(ps):
    f1 = lambda p: -Epf(p, ps)
    p = optimize.minimize_scalar(f1, bounds=(-1000, ps), method='bounded').x
    return p

def gcmaxf(ps, a=1.6, D=0.005):
    p = pminf(ps)
    return (ps-p)*kf((ps+p)/2)/a/D
    

def pf(gc, ps):
    if gc == gcmaxf(ps):
        return pminf(ps)
    f1 = lambda p: Egcf(gc)-Epf(p, ps)
    pmin = pminf(ps)
    p = optimize.brentq(f1, pmin, ps)
    return p

def net_gainf(gc, ps, kmax=0.01):
    p = pf(gc, ps)
    k_cost = kf(p)/kmax
    A = Af(gc)
    return A*k_cost

# simulation
def simf(gcpsf, r0=0.3, r1=0, t0=10, t1=30, dt=60*60,
         l=1.8*10**-5, n=0.43, Z=3):
    t0, t1 = t0*24, t1*24
    t_total = t0+t1
    r = np.zeros(t_total)
    r[0] = r0
    r[t0] = r1
    s = np.zeros(t_total+1)
    net_gain = np.zeros(t_total)
    ps = np.zeros(t_total)
    gc = np.zeros(t_total)
    E = np.zeros(t_total)
    for i in range(t_total):
        s[i] = min(1, s[i]+r[i])
        ps[i] = psf(s[i])
        gc[i] = gcpsf(psf(s[i]))
        net_gain[i] = net_gainf(gc[i], ps[i])*dt
        E[i] = Egcf(gc[i])*dt*l/n/Z
        s[i+1] = s[i]-E[i]
    return sum(net_gain)/t_total/dt*10**6 # μmol m-2 s-1 MPa-1

# the same simulation with different outputs
def simf1(gcpsf, r0=0.3, r1=0, t0=100, t1=1, dt=60*60,
          l=1.8*10**-5, n=0.43, Z=3):
    t0, t1 = t0*24, t1*24
    t_total = t0+t1
    r = np.zeros(t_total)
    r[0] = r0
    r[t0] = r1
    s = np.zeros(t_total+1)
    net_gain = np.zeros(t_total)
    ps = np.zeros(t_total)
    gc = np.zeros(t_total)
    E = np.zeros(t_total)
    for i in range(t_total):
        s[i] = min(1, s[i]+r[i])
        ps[i] = psf(s[i])
        gc[i] = gcpsf(psf(s[i]))
        net_gain[i] = net_gainf(gc[i], ps[i])*dt
        E[i] = Egcf(gc[i])*dt*l/n/Z
        s[i+1] = s[i]-E[i]
    return s, ps, gc, net_gain/dt*10**6, E

# expected net carbon gain
def encgf(gcpsf, freq, MAP, s_min=0.05,
          a=1.6, D=0.005, l=1.8*10**-5, n=0.43, Z=3):
    MDP = MAP/365
    gamma = 1/((MDP/freq)/1000)*n*Z
    
    Lf = lambda s: a*D*gcpsf(psf(s))*l/n/Z+s/1000
    rLf = lambda s: 1/Lf(s)
    integralf = lambda s: integrate.quad(rLf, s, 1)[0]
    pdf_no_c = lambda s: 1/Lf(s)*np.exp(-gamma*s-freq*integralf(s))
    
    c = 1/(integrate.quad(pdf_no_c, s_min, 1)[0])
    pdf = lambda s: c*pdf_no_c(s)
    f1 = lambda s: net_gainf(gcpsf(psf(s)), psf(s))*pdf(s)
    return integrate.quad(f1, s_min, 1)[0]*10**6
