from Functions import ESSf, gcsLf
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

pk, kmax, a, D, l, n, Z = 0.5, 0.01, 1.6, 0.005, 1.8*10**-5, 0.43, 3
freq1, MAP1 = 0.05, 1000
freq2, MAP2 = 0.05, 3000
sL1 = optimize.brentq(ESSf, 0.1, 0.5, args=(freq1, MAP1,\
                                            pk, kmax, a, D, l, n, Z))
sL2 = optimize.brentq(ESSf, 0.1, 0.5, args=(freq2, MAP2,\
                                            pk, kmax, a, D, l, n, Z))

if __name__ == "__main__":
    s1 = np.linspace(sL1, 1, 100)
    s2 = np.linspace(sL2, 1, 100)
    gc1 = [gcsLf(i, sL1, pk, kmax, a, D) for i in s1]
    gc2 = [gcsLf(i, sL1, pk, kmax, a, D) for i in s2]
    plt.plot(s1, gc1, label='freq={}, MAP={}'.format(freq1, MAP1))
    plt.plot(s2, gc2, color='red', label='freq={}, MAP={}'.format(freq2, MAP2))