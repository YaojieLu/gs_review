from Functions import gcmaxf, encgf
from eller import eller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

def f(freq, MAP):
    # objective function
    def objf(a, b, c):
        gcpsf = lambda ps: min(a*np.exp(-(ps/c)**b), gcmaxf(ps))
        try:
            res = encgf(gcpsf, freq, MAP)
        except ValueError:
            res = 0
        return res

    # parameter space
    pbounds = {'a': (0, 1), 'b': (0, 100), 'c': (-10, -0.01)}
    
    # bayesian optimization
    optimizer = BayesianOptimization(
        f=objf,
        pbounds=pbounds,
        verbose=1,
        random_state=11
        )
    
    # # prior
    # optimizer.probe(
    #     params={'a': 0.4, 'b': 1, 'c': -1},
    #     lazy=True
    #     )
    
    # reiterate
    optimizer.set_bounds(new_bounds={'a': (0, 1)})
    optimizer.maximize(
        init_points=2,
        n_iter=100,
        )
    return optimizer.max['target']

# run
freq = [0.025, 0.05]
MAP = np.linspace(500, 3000, 6)
df = pd.DataFrame([[x, y] for x in freq for y in MAP],\
                   columns=['freq', 'MAP'])
df['encgf'] = df.apply(lambda row: f(row['freq'], row['MAP']), axis=1)
df.to_csv('Stochastic-rainfall-long-term.csv')