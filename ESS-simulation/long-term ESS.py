from Functions import simrif
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

def best_invaderf(a, b, c):
    # objective function
    def objf(ai, bi, ci):
        return simrif(a, b, c, ai, bi, ci)
    
    # parameter space
    pbounds = {'ai': (0, 1), 'bi': (0, 2), 'ci': (-10, -0.01)}
    
    # bayesian optimization
    optimizer = BayesianOptimization(
        f=objf,
        pbounds=pbounds,
        verbose=0,
        random_state=1
        )
    
    # reiterate
    optimizer.maximize(
        init_points=3,
        n_iter=50,
        )
    
    # resident vs. invader strategy
    ai, bi, ci = optimizer.max['params'].values()
    gcpsf = lambda ps, a, b, c: a*np.exp(-(ps/c)**b)
    s = np.linspace(-1, 0.1, 10)
    gcr = [gcpsf(i, a, b, c) for i in s]
    gci = [gcpsf(i, ai, bi, ci) for i in s]
    return max([abs(i-j) for i, j in zip(gcr, gci)])

def ESSf():
    # objective function
    def objf(a, b, c):
        return -best_invaderf(a, b, c)
    
    # parameter space
    pbounds = {'a': (0, 1), 'b': (0, 2), 'c': (-10, -0.01)}
    
    # bayesian optimization
    optimizer = BayesianOptimization(
        f=objf,
        pbounds=pbounds,
        verbose=1,
        random_state=1
        )
    
    # reiterate
    optimizer.maximize(
        init_points=3,
        n_iter=50,
        )
    
    return optimizer.max['params'].values()
    
# run
a, b, c = ESSf()
# 0.7010311126535608, 0.061750783232156214, -9.10682538294527