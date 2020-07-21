from Functions import simESSf
import matplotlib.pyplot as plt

a, b, c = 0.7010311126535608, 0.061750783232156214, -9.10682538294527
PLCr, s = simESSf(a, b, c, a, b, c)


# figure
fig, ax = plt.subplots()
ax.plot(PLCr[0::24], color='black')
ax.set_xlabel('Days', fontsize=14)
ax.set_ylabel('Permanent PLC', fontsize=14)
ax.set_ylim(0, 1)
ax2=ax.twinx()
ax2.plot(s[:-1][0::24], color='blue')
ax2.set_ylabel('Relative soil water availability', color='blue', fontsize=14)
ax2.set_ylim(0, 1)