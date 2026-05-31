import numpy as np
from scipy.optimize import minimize

def prior_from_returns(returns, n_scenarios=500):
    """
    Generate prior scenarios (N x K matrix) and equal probabilities.
    """
    rets = returns.values
    T, K = rets.shape
    if T < 2:
        # Not enough data, return trivial prior
        scenarios = np.zeros((n_scenarios, K))
        p_prior = np.ones(n_scenarios) / n_scenarios
        return scenarios, p_prior
    if T < n_scenarios:
        n_scenarios = T
    idx = np.random.choice(T, n_scenarios, replace=True)
    scenarios = rets[idx]
    p_prior = np.ones(n_scenarios) / n_scenarios
    return scenarios, p_prior

def entropy_pooling(p_prior, scenarios, view_matrices, view_targets, confidence=0.7, eps=1e-12):
    """
    Returns (mu_post, p_post)
    """
    n_scenarios = len(p_prior)
    # Ensure prior probabilities are positive
    p_prior = np.maximum(p_prior, eps)
    p_prior = p_prior / np.sum(p_prior)
    
    # Build constraints: A_eq @ p = b_eq
    A_eq_list = []
    b_eq_list = []
    
    # Add view constraints
    for V, target in zip(view_matrices, view_targets):
        # V: shape (1, K) row vector
        A_row = (V @ scenarios.T).flatten()  # shape (n_scenarios,)
        A_eq_list.append(A_row)
        b_eq_list.append(target)
    
    # Add sum constraint
    A_eq_list.append(np.ones(n_scenarios))
    b_eq_list.append(1.0)
    
    A_eq = np.vstack(A_eq_list)
    b_eq = np.array(b_eq_list)
    
    # Objective and gradient
    def objective(p):
        p = np.maximum(p, eps)
        return np.sum(p * np.log(p / p_prior))
    
    def jac(p):
        p = np.maximum(p, eps)
        return 1 + np.log(p / p_prior)
    
    # Constraints: p >= 0, A_eq p = b_eq
    constraints = [{'type': 'eq', 'fun': lambda p: A_eq @ p - b_eq}]
    bounds = [(eps, None)] * n_scenarios
    
    # Initial guess: prior
    p0 = p_prior.copy()
    
    # Solve
    try:
        res = minimize(objective, p0, jac=jac, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-6, 'maxiter': 500})
        if res.success:
            p_post = res.x
        else:
            p_post = p_prior
    except:
        p_post = p_prior
    
    # Posterior expected returns
    mu_post = scenarios.T @ p_post
    return mu_post, p_post

def compute_pooled_scores(returns, engine_views=None, confidence=0.7, n_scenarios=500):
    """
    Returns dict of expected returns per ticker.
    """
    K = returns.shape[1]
    if K < 2:
        return {returns.columns[0]: 0.0}
    
    scenarios, p_prior = prior_from_returns(returns, n_scenarios)
    view_matrices = []
    view_targets = []
    
    if engine_views is None:
        # Default momentum view
        last_ret = returns.iloc[-1].values
        # Top 3 by last return
        top3_idx = np.argsort(last_ret)[-3:]
        V = np.zeros(K)
        V[top3_idx] = 1.0 / 3.0
        market_mean = np.mean(last_ret)
        target = market_mean + 0.002  # 0.2% above mean
        view_matrices.append(V.reshape(1, -1))
        view_targets.append(target)
    else:
        for weights, target in engine_views:
            view_matrices.append(np.array(weights).reshape(1, -1))
            view_targets.append(target)
    
    mu_post, _ = entropy_pooling(p_prior, scenarios, view_matrices, view_targets, confidence)
    return {ticker: mu_post[i] for i, ticker in enumerate(returns.columns)}
