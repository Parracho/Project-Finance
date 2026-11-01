import statsmodels.api as sm
import pandas as pd
import numpy as np



def t_stat(coefs):
    results={}
    for s in coefs.columns:
        if coefs[s].dropna().shape[0] > 48:
            reg=pd.concat({"y": coefs[s].shift(-1), "x": coefs[s]}, axis=1).dropna()
            X = sm.add_constant(reg["x"])
            ar = sm.OLS(reg["y"],X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            
            lam = coefs[s].dropna()
            mu = sm.OLS(lam.values, np.ones(len(lam))).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
            results[s] = {
                "mean": np.asarray(mu.params)[0],        # average daily payoff
                "t_mean": np.asarray(mu.tvalues)[0],     # does it pay? (Fama-MacBeth test)
                "rho": ar.params["x"],                   # persistence
                "t_rho": ar.tvalues["x"],
                "R2": ar.rsquared,
                "n": int(ar.nobs),
            }
    return pd.DataFrame(results).T