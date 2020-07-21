from Functions import *#simf
import matplotlib.pyplot as plt

sLR, s0, t_total, dt = 0.3, 1, 100, 60*60
pk, kmax, a, D, l, n, Z = 0.5, 0.01, 1.6, 0.005, 1.8*10**-5, 0.43, 3
PLCI, net_gainI, net_gainR = simf(sLR, s0, t_total, dt, pk, kmax, a, D, l, n, Z)
plt.plot(PLCI)