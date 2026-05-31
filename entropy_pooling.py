import numpy as np
from scipy.optimize import minimize

def prior_from_returns(returns, n_scenarios=500):
    """
    Generate prior scenarios (N x K matrix) and equal probabilities.
    Uses bootstrap with replacement.
    """
    rets = returns.values
    if len(rets) < 2:
        n_scenarios = len(rets)
    if n_scenarios < 1:
        return np.zeros((1, rets.shape[1])), np.array([1.0])
    idx = np.random.choice(len(rets), min(n_scenarios, len(rets)), replace=True)
    scenarios = rets[idx]
    p_prior = np.ones(len(scenarios)) / len(scenarios)
    return scenarios, p_prior

def entropy_pooling(p_prior, scenarios, view_matrices, view_targets, confidence=0.7, eps=1e-12):
    """
    Minimum relative entropy: find p that minimizes KL(p || p_prior)
    subject to sum(p) = 1, p >= 0, and view constraints: V * (scenarios' * p) = targets.
    """
    n_scenarios = len(p_prior)
    # Add small epsilon to prior to avoid zeros
    p_prior = p_prior + eps
    p_prior = p_prior / np.sum(p_prior)
    
    # Build equality constraints: A_eq @ p = b_eq
    A_eq_list = []
    b_eq_list = []
    for V, target in zip(view_matrices, view_targets):
        V = np.asarray(V).reshape(1, -1)
        A_row = (V @ scenarios.T).reshape(1, -1)
        A_eq_list.append(A_row)
        b_eq_list.append([target])
    if A_eq_list:
        A_eq = np.vstack(A_eq_list)
        b_eq = np.hstack(b_eq_list)
    else:
        A_eq = None
        b_eq = None
    # Also constraint sum(p) = 1
    A_sum = np.ones((1, n_scenarios))
    b_sum = np.array([1.0])
    if A_eq is not None:
        A_eq = np.vstack([A_eq, A_sum])
        b_eq = np.hstack([b_eq, b_sum])
    else:
        A_eq = A_sum
        b_eq = b_sum
    
    def objective(p):
        p = np.maximum(p, eps)
        return np.sum(p * np.log(p / p_prior))
    
    def jac(p):
        p = np.maximum(p, eps)
        return 1 + np.log(p / p_prior)
    
    constraints = [{'type': 'eq', 'fun': lambda p: A_eq @ p - b_eq}]
    bounds = [(eps, None)] * n_scenarios
    
    p0 = p_prior.copy()
    res = minimize(objective, p0, jac=jac, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-6, 'maxiter': 1000})
    if res.success:
        p_post = res.x
    else:
        p_post = p_prior
    
    mu_post = scenarios.T @ p_post
    return mu_post, p_post

def compute_pooled_scores(returns, engine_views=None, confidence=0.7, n_scenarios=500):
    """
    Generate prior from returns, then incorporate views.
    Returns a tuple (score_dict, None) for compatibility with train.py that expects two values.
    """
    if returns.shape[1] == 0 or returns.shape[0] < 2:
        scores = {ticker: 0.0 for ticker in returns.columns}
        return scores, None
    K = returns.shape[1]
    scenarios, p_prior = prior_from_returns(returns, n_scenarios)
    
    view_matrices = []
    view_targets = []
    if engine_views is None:
        # Simple momentum view: top 3 ETFs by last return outperform by 0.2%
        last_ret = returns.iloc[-1].values
        top3_idx = np.argsort(last_ret)[-3:]
        V = np.zeros(K)
        V[top3_idx] = 1.0 / 3.0
        market_mean = np.mean(last_ret)
        target = market_mean + 0.002
        view_matrices.append(V.reshape(1, -1))
        view_targets.append(target)
    else:
        for weights, target in engine_views:
            view_matrices.append(np.array(weights).reshape(1, -1))
            view_targets.append(target)
    
    mu_post, _ = entropy_pooling(p_prior, scenarios, view_matrices, view_targets, confidence)
    # Ensure all values are floats, replace NaN with 0.0
    scores = {ticker: float(mu_post[i]) if not np.isnan(mu_post[i]) else 0.0 for i, ticker in enumerate(returns.columns)}
    return scores, None
