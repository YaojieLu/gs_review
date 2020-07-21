from Functions import simf, simf1
from eller import eller
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

def f(r0=0.3, r1=0, t0=100, t1=1):
    # objective function
    def objf(a, b, c):
        gcpsf = lambda ps: a*np.exp(-(ps/c)**b)
        try:
            res = simf(gcpsf, r0=r0, r1=r1, t0=t0, t1=t1)
        except ValueError:
            res = 0
        return res

    # parameter space
    pbounds = {'a': (0, 1), 'b': (0, 10), 'c': (-1, -0.01)}
    
    # bayesian optimization
    optimizer = BayesianOptimization(
        f=objf,
        pbounds=pbounds,
        verbose=1,
        random_state=1
        )
    
    # # prior
    # optimizer.probe(
    #     params={'a': 0.4, 'b': 1, 'c': -1},
    #     lazy=True
    #     )
    
    # # reiterate
    # optimizer.set_bounds(new_bounds={'a': (0, 1)})
    optimizer.maximize(
        init_points=10,
        n_iter=100,
        )
    return optimizer.max

if __name__ == "__main__":
    r0, r1, t0, t1 = 0.3, 0, 100, 1
    lt = f(r0=r0, r1=r1, t0=t0, t1=t1)
    a, b, c = lt['params'].values()
    ltf = lambda ps, a=a, b=b, c=c: a*np.exp(-(ps/c)**b)
    s1, ps1, gc1, net_gain1, E1 = simf1(eller, r0=r0, r1=r1, t0=t0, t1=t1)
    s2, ps2, gc2, net_gain2, E2 = simf1(ltf, r0=r0, r1=r1, t0=t0, t1=t1)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(net_gain1, label='instantaneous optimization')
    ax[0].plot(net_gain2, color='red', label='long-term optimization')
    ax[0].set_xlabel('Time (hr)', fontsize=20)
    ax[0].set_ylabel('net carbon gain rate (μmol m-2 s-1 MPa-1)', fontsize=20)
    ax[0].legend(fontsize=20)
    ax[1].plot(s1, label='instantaneous optimization')
    ax[1].plot(s2, color='red', label='long-term optimization')
    ax[1].set_xlabel('Time (hr)', fontsize=20)
    ax[1].set_ylabel('relative soil water availability', fontsize=20)
    ax[1].set_ylim([0, 0.35])
