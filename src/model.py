# Credit limit optimizer — assigns limits based on PD thresholds
import numpy as np
from src.core import profit_at_limit

def fit_and_evaluate(data, seed=42):
    df = data["df"]
    limits = []
    for _, row in df.iterrows():
        base = row["current_limit"]
        if row["pd"] < 0.05:
            new = base * 1.5
        elif row["pd"] < 0.15:
            new = base * 1.0
        else:
            new = base * 0.5
        limits.append(max(500, min(100000, int(new))))
    
    profits = [profit_at_limit(limits[i], df.iloc[i]["pd"]) for i in range(len(limits))]
    return (
        {"optimized_limits": limits},
        {
            "n_customers": len(limits),
            "total_profit": float(np.sum(profits)),
            "avg_profit": float(np.mean(profits)),
        },
    )