# Credit limit P&L helpers — no ML, just business math
import numpy as np

def profit_at_limit(limit, loss_rate, interest_rate=0.15, funding_cost=0.03, expected_loss=0.04):
    rev = limit * interest_rate
    cost = limit * funding_cost
    loss = limit * expected_loss
    return float(rev - cost - loss)

def constraint_satisfaction(limits, min_limits, max_limits):
    ok = 0
    for l, mi, ma in zip(limits, min_limits, max_limits):
        if mi <= l <= ma:
            ok += 1
    return ok / len(limits) if limits else 1.0