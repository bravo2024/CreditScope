# Synthetic credit limit data: income, FICO, balance, current limit, PD
import numpy as np
import pandas as pd

def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(11, 0.5, n).clip(20000, 500000).astype(int)
    fico = rng.integers(500, 800, n)
    balance = rng.lognormal(8, 1, n).astype(int)
    current_limit = np.clip(balance * 2 + rng.integers(-1000, 5000, n), 1000, 50000).astype(int)
    logit = -3 - 0.01 * fico + 0.5 * np.log(balance / 1000)
    prob_def = 1.0 / (1.0 + np.exp(-logit))
    df = pd.DataFrame({
        "income": income, "fico": fico, "balance": balance,
        "current_limit": current_limit, "pd": prob_def,
    })
    return {"df": df, "n_samples": n}