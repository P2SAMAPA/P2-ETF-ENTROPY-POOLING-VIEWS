import numpy as np
from scipy.optimize import root

def prior_from_returns(returns, n_scenarios=500):
    rets = returns.values
    if len(rets) < n_scenarios:
        n_scenarios = len(rets)
    # Use bootstrap
    idx = np.random.choice(len(rets), n_scenarios, replace=True)
    scenarios = rets[idx]
    p_prior = np.ones(n_scenarios) / n_scenarios
    return scenarios, p_prior

def exponential_tilting(p_prior, scenarios, view_matrices, view_targets, confidence=0.7):
    """
    Solve for Lagrange multipliers eta such that:
        sum_i p_prior_i * exp( eta' @ (V @ scenario_i) ) * (V @ scenario_i) = target
    Then posterior probabilities: p_i = p_prior_i * exp( eta' @ (V @ scenario_i) )
    """
    n_scenarios = len(p_prior)
    n_views = len(view_matrices)
    # Compute view values for each scenario: v_i = V @ scenario_i (a vector of length n_views)
    view_values = np.zeros((n_scenarios, n_views))
    for v_idx, (V, _) in enumerate(view_matrices):
        view_values[:, v_idx] = (V @ scenarios.T).flatten()
    # Define function to find eta
    def equations(eta):
        # eta shape (n_views,)
        exp_term = np.exp(view_values @ eta)
        weights = p_prior * exp_term
        weights = weights / np.sum(weights)  # normalise
        # Expected view values under posterior
        expected = weights @ view_values
        # Target
        targets = np.array(view_targets)
        return expected - targets
    # Initial guess
    eta0 = np.zeros(n_views)
    try:
        sol = root(equations, eta0, method='hybr')
        eta = sol.x
    except:
        eta = eta0
    # Compute posterior probabilities
    exp_term = np.exp(view_values @ eta)
    p_post = p_prior * exp_term
    p_post = p_post / np.sum(p_post)
    # Posterior expected returns
    mu_post = scenarios.T @ p_post
    return mu_post, p_post

def compute_pooled_scores(returns, engine_views=None, confidence=0.7, n_scenarios=500):
    K = returns.shape[1]
    scenarios, p_prior = prior_from_returns(returns, n_scenarios)
    view_matrices = []
    view_targets = []
    if engine_views is None:
        last_ret = returns.iloc[-1].values
        top3_idx = np.argsort(last_ret)[-3:]
        V = np.zeros(K)
        V[top3_idx] = 1.0 / 3.0
        market_mean = np.mean(last_ret)
        target = market_mean + 0.002 * confidence  # scale by confidence
        view_matrices.append(V.reshape(1, -1))
        view_targets.append(target)
    else:
        for weights, target in engine_views:
            view_matrices.append(np.array(weights).reshape(1, -1))
            view_targets.append(target)
    mu_post, _ = exponential_tilting(p_prior, scenarios, view_matrices, view_targets, confidence)
    # Ensure no NaN
    mu_post = np.nan_to_num(mu_post, nan=0.0, posinf=0.0, neginf=0.0)
    return {ticker: float(mu_post[i]) for i, ticker in enumerate(returns.columns)}
