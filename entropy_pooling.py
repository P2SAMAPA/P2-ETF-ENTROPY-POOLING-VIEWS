import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

def prior_from_returns(returns, n_scenarios=500):
    """
    Generate prior scenarios (N x K matrix) and equal probabilities.
    Uses kernel density estimation to simulate from empirical distribution.
    """
    rets = returns.values
    if len(rets) < n_scenarios:
        n_scenarios = len(rets)
    # Use bootstrap – random samples with replacement
    idx = np.random.choice(len(rets), n_scenarios, replace=True)
    scenarios = rets[idx]  # shape (n_scenarios, K)
    p_prior = np.ones(n_scenarios) / n_scenarios
    return scenarios, p_prior

def entropy_pooling(p_prior, scenarios, view_matrices, view_targets, confidence=0.7, eps=1e-12):
    """
    Minimum relative entropy: find p that minimizes KL(p || p_prior)
    subject to sum(p) = 1, p >= 0, and view constraints: V * (scenarios' * p) = targets.
    """
    n_scenarios = len(p_prior)
    n_assets = scenarios.shape[1]
    
    # Add small epsilon to prior probabilities to avoid zeros
    p_prior = p_prior + eps
    p_prior = p_prior / np.sum(p_prior)
    
    # Build equality constraints: A_eq @ p = b_eq
    A_eq_list = []
    b_eq_list = []
    for V, target in zip(view_matrices, view_targets):
        # V: (1 x K) row vector
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
    
    # Objective: KL divergence = sum p_i log(p_i / p_prior_i)
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
    
    res = minimize(objective, p0, jac=jac, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-6, 'maxiter': 1000})
    if res.success:
        p_post = res.x
    else:
        # fallback: prior
        p_post = p_prior
    
    # Posterior expected returns
    mu_post = scenarios.T @ p_post
    return mu_post, p_post

def compute_pooled_scores(returns, engine_views=None, confidence=0.7, n_scenarios=500):
    """
    Generate prior from returns, then incorporate views.
    If engine_views is None, we create a simple momentum view: top 3 ETFs by last return outperform.
    """
    K = returns.shape[1]
    scenarios, p_prior = prior_from_returns(returns, n_scenarios)
    # Build views
    view_matrices = []
    view_targets = []
    if engine_views is None:
        # Simple momentum view: last day's return
        last_ret = returns.iloc[-1].values
        # View: ETFs with positive last return will have expected return 0.5% above market mean
        # For simplicity, we use a relative ranking: top 3 outperform by 0.2%
        top3_idx = np.argsort(last_ret)[-3:]
        # View: expected return of top3 = market mean + 0.002
        V = np.zeros(K)
        V[top3_idx] = 1.0 / 3.0
        market_mean = np.mean(last_ret)
        target = market_mean + 0.002
        view_matrices.append(V.reshape(1, -1))
        view_targets.append(target)
    else:
        # Use provided engine views: each view is (weights, target)
        for weights, target in engine_views:
            view_matrices.append(np.array(weights).reshape(1, -1))
            view_targets.append(target)
    mu_post, p_post = entropy_pooling(p_prior, scenarios, view_matrices, view_targets, confidence)
    return {ticker: mu_post[i] for i, ticker in enumerate(returns.columns)}
