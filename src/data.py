from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["credit_score", "income", "loan_amount", "dti_ratio", "employment_months", "num_credit_lines", "delinquencies_2yr", "revolving_util", "inquiries_6mo", "age", "home_owner", "education"]
CATEGORICAL_FEATURES = ["home_owner", "education"]
NUMERICAL_FEATURES = ["credit_score", "income", "loan_amount", "dti_ratio", "employment_months", "num_credit_lines", "delinquencies_2yr", "revolving_util", "inquiries_6mo", "age"]

def make_synthetic(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "credit_score": rng.normal(700, 70, size=n).clip(300, 850).astype(int),
        "income": rng.lognormal(mean=11.2, sigma=0.5, size=n).clip(20000, 500000).astype(int),
        "loan_amount": rng.lognormal(mean=10.0, sigma=0.7, size=n).clip(1000, 50000).astype(int),
        "dti_ratio": rng.uniform(5, 45, size=n).round(1),
        "employment_months": rng.gamma(shape=4, scale=15, size=n).clip(0, 480).astype(int),
        "num_credit_lines": rng.poisson(lam=8, size=n).clip(1, 30),
        "delinquencies_2yr": rng.poisson(lam=0.5, size=n).clip(0, 8),
        "revolving_util": rng.uniform(0, 100, size=n).round(1),
        "inquiries_6mo": rng.poisson(lam=1.0, size=n).clip(0, 10),
        "age": rng.normal(38, 12, size=n).clip(18, 75).astype(int),
        "home_owner": rng.choice(["yes", "no", "rent"], size=n, p=[0.45, 0.30, 0.25]),
        "education": rng.choice(["high_school", "bachelors", "masters", "phd"], size=n, p=[0.30, 0.40, 0.22, 0.08]),
    })
    cs = (df["credit_score"] - 300) / 550; dti = df["dti_ratio"] / 45; delinq = np.clip(df["delinquencies_2yr"], 0, 3) / 3
    util = df["revolving_util"] / 100; emp = np.clip(df["employment_months"] / 120, 0, 1)
    income = np.log(df["income"] / 1000) / 5
    log_odds = -2.0 + 2.0 * cs - 0.8 * dti - 0.5 * delinq - 0.6 * util + 0.3 * emp + 0.3 * income + rng.normal(0, 0.5, size=n)
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > np.percentile(prob, 80)).astype(np.float64)
    return {"X": df, "y": y, "features": FEATURE_NAMES, "df": df.assign(default=y), "categorical_features": CATEGORICAL_FEATURES, "numerical_features": NUMERICAL_FEATURES, "n_samples": n, "n_features": len(FEATURE_NAMES), "positive_rate": y.mean()}
