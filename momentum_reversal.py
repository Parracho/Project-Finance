import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path


def momentum_reversal(k_max, ret):

    # -------------------------------------------------
    # 3. Cross-sectional regressions
    # -------------------------------------------------

    betas = {}

    for k in range(1, k_max + 1):

        # r(i,t-k)
        ret_lag = ret.shift(k)

        # Need at least k previous observations
        for t in range(k + 1, len(ret)):

            # Current cross-section: r(i,t)
            y_t = ret.iloc[t]

            # Lagged cross-section: r(i,t-k)
            x_t = ret_lag.iloc[t]

            # Combine and remove missing values
            regression_data = pd.concat(
                [y_t.rename('y'), x_t.rename('x')],
                axis=1
            ).dropna()

            # Skip if insufficient assets
            if len(regression_data) < 3:
                continue

            # r(i,t) = alpha(k,t) + gamma(k,t) * r(i,t-k) + u(i,t)
            X = sm.add_constant(regression_data['x'])
            model = sm.OLS(regression_data['y'], X).fit()

            betas[(ret.index[t], k)] = {
                'alpha': model.params['const'],
                'gamma': model.params['x'],
                't_stat': model.tvalues['x'],
                'n_obs': len(regression_data)
            }

    # -------------------------------------------------
    # 4. Average gammas across time
    # -------------------------------------------------

    results = []


    for k in range(1, k_max + 1):

        gammas = [
            values['gamma']
            for (t, lag), values in betas.items()
            if lag == k
        ]


        if len(gammas) > 0:

            mean_gamma = np.mean(gammas)

            results.append({
                'lag': k,
                'mean_gamma': mean_gamma,
                'n_periods': len(gammas)
            })

    results = pd.DataFrame(results).set_index('lag')

    # -------------------------------------------------
    # 5. Plot gammas_mean.abs().sum()
    # -------------------------------------------------
    gammas_mean_normalized=results['mean_gamma']/results['mean_gamma'].abs().sum()
    gammas_mean_normalized.plot(
        figsize=(10, 5),
        kind='bar'
    )

    plt.xlabel('Lag k (30-minute periods)')
    plt.ylabel('Average gamma')
    plt.title('Average Cross-Sectional Return Response by Lag')
    plt.show()

    return betas, ret