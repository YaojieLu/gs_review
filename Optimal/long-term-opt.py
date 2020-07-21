from Functions import simf
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
    
    # reiterate
    optimizer.set_bounds(new_bounds={'a': (0, 1)})
    optimizer.maximize(
        init_points=10,
        n_iter=100,
        )
    return optimizer.max['target'], simf(eller, r0=r0, r1=r1, t0=t0, t1=t1)

# run
x = np.linspace(0.2, 1, 9)
y = [f(r0=i, r1=0, t0=100, t1=1) for i in x]

# figure
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(x, np.array(y)[:,0], label='long-term optimzation')
ax[0].plot(x, np.array(y)[:,1], color='red', label='instantaneous optimization')
ax[0].set_xlim([0, 1])
ax[0].set_xlabel('initial soil water availability', fontsize=20)
ax[0].set_ylabel('mean net carbon gain rate (μmol m-2 s-1 MPa-1)', fontsize=20)
ax[0].legend(fontsize=20)
ax[1].plot(x, np.array(y)[:,1]/np.array(y)[:,0]*100, 'o')
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 105])
ax[1].set_xlabel('initial soil water availability', fontsize=20)
ax[1].set_ylabel('instantaneous/long-term net carbon gain rate (%)', fontsize=20)
