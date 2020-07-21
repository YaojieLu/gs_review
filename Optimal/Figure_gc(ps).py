from Functions import gcmaxf, net_gainf
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def eller(ps):
    gcmax = gcmaxf(ps)
    f1 = lambda gc: -net_gainf(gc, ps)
    gc = optimize.minimize_scalar(f1, bounds=(0, gcmax), method='bounded')
    return gc.x

gcpsf = lambda ps, a, b, c: a*np.exp(-(ps/c)**b)

if __name__ == "__main__":
    x = np.linspace(-2, 0, 100)
    y = [eller(i) for i in x]
    y2 = [gcpsf(i, a=0.18, b=0.435, c=-0.26) for i in x]
    plt.plot(x, y, label='eller')
    plt.plot(x, y2, color='red', label='test')