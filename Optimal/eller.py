from Functions import gcmaxf, net_gainf, simf
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def eller(ps):
    gcmax = gcmaxf(ps)
    f1 = lambda gc: -net_gainf(gc, ps)
    gc = optimize.minimize_scalar(f1, bounds=(0, gcmax), method='bounded')
    return gc.x

if __name__ == "__main__":
    total_net_gain = simf(eller)
    print(total_net_gain)
    
    x = np.linspace(-2, 0, 100)
    y = [eller(i) for i in x]
    y2 = [0.4*np.exp(i) for i in x]
    plt.plot(x, y)
    plt.plot(x, y2, color='red')