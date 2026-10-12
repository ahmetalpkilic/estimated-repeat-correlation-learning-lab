"""Temporal calibration-repeat correlation -> covariance -> optical delay."""
import numpy as np

LABELS = np.arange(40) % 2


def correlation(m, rho):
    """Stationary AR(1) correlation between repeats, not between clocks."""
    if not isinstance(m, int) or m < 3 or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('Require integer m>=3 and finite |rho|<1')
    return rho ** np.abs(np.arange(m)[:, None] - np.arange(m))


def scatter_factor(m, rho):
    """E[ordinary sample covariance] = b*C, using tr(H R)/(m-1)."""
    r = correlation(m, rho)
    return (m - r.sum()/m)/(m-1)


def residual_precision(m, rho):
    """GLS removes an unknown constant event mean in the R^-1 metric."""
    r = correlation(m, rho)
    w = np.linalg.solve(r, np.eye(m))
    v = w @ np.ones(m)
    return w - np.outer(v, v)/v.sum()


def estimate(repeats, rhos, method='gls'):
    """Estimate marginal one-pair C, then pool within known identical classes.

    GLS scatter has m-1 residual degrees of freedom when R is known. Subtracting
    an ordinary mean first is harmless because Q annihilates constant vectors.
    The scalar correction is unbiased but does not whiten temporal modes.
    """
    z = np.asarray(repeats, float)
    rhos = np.asarray(rhos, float)
    if z.ndim != 3 or z.shape[0] != 40 or z.shape[1] < 3 or z.shape[2] != 2 or not np.all(np.isfinite(z)) or rhos.shape != (40,):
        raise ValueError('Require finite (40,m>=3,2) repeats and 40 correlations')
    if method not in ('naive', 'scalar', 'gls'):
        raise ValueError('Unknown estimator')
    m = z.shape[1]
    s = []
    for event, rho in zip(z, rhos):
        correlation(m, float(rho))  # validate even for the naive control
        residual = event-event.mean(axis=0)
        if method == 'gls':
            cov = residual.T @ residual_precision(m, float(rho)) @ residual/(m-1)
        else:
            cov = residual.T @ residual/(m-1)
            if method == 'scalar': cov /= scatter_factor(m, float(rho))
        s.append(cov)
    s = np.asarray(s)
    pooled = np.stack([s[LABELS == g].mean(axis=0) for g in LABELS])
    if np.any(np.linalg.eigvalsh(pooled) <= 0):
        raise ValueError('Singular covariance estimate')
    return pooled


def draw_repeats(normal, cov, rhos, means):
    """Z_i = 1 mu_i^T + L_R E_i L_C^T (separable matrix-normal noise)."""
    lr = np.stack([np.linalg.cholesky(correlation(normal.shape[1], float(r))) for r in rhos])
    lc = np.linalg.cholesky(cov)
    return means[:, None] + np.einsum('nkl,nlj,nij->nki', lr, normal, lc)


