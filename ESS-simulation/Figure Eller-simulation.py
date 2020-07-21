from Functions import simf
import matplotlib.pyplot as plt

s0, t = 1, 100
a, D, l, n, Z = 1.6, 0.005, 1.8*10**-5, 0.43, 0.8
PLC, s = simf(s0, t, a, D, l, n, Z)


# figure
fig, ax = plt.subplots()
ax.plot(PLC[0::24], color='black')
ax.set_xlabel('Days', fontsize=14)
ax.set_ylabel('Permanent PLC', fontsize=14)
ax.set_ylim(0, 1)
ax2=ax.twinx()
ax2.plot(s[:-1][0::24], color='blue')
ax2.set_ylabel('Relative soil water availability', color='blue', fontsize=14)
ax2.set_ylim(0, 1)