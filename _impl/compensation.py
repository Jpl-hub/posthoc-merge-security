"""Finite minimax compensation with full native and projected public outputs."""
import numpy as np
import torch
from scipy.optimize import minimize


def compensate(K, full_native, projected, group_ids, Q, B, rho=0.01):
    """One convex conditional B problem; group normalization is supplied by caller."""
    n = full_native.shape[1]
    groups = int(group_ids.max()) + 1
    free = full_native - Q @ (Q.T @ full_native)
    ridge = rho / float(B.square().sum())
    trace = []

    def solve(weights):
        lam = torch.as_tensor(np.maximum(weights, 0), dtype=K.dtype, device=K.device)
        scale = lam[group_ids].sqrt()
        kw, ew = K * scale, projected * scale
        gram = kw.T @ kw
        gram.diagonal().add_(ridge)
        parallel = torch.cholesky_solve(ew.T, torch.linalg.cholesky(gram)).T @ kw.T
        kn, en = kw[:, :n], free * scale[:n]
        gram_native = kn.T @ kn
        gram_native.diagonal().add_(ridge)
        other = torch.cholesky_solve(en.T, torch.linalg.cholesky(gram_native)).T @ kn.T
        D = Q @ parallel + other
        residual = (parallel @ K - projected).square().sum(0)
        residual[:n] += (other @ K[:, :n] - free).square().sum(0)
        errors = torch.zeros(groups, dtype=K.dtype, device=K.device).scatter_add_(
            0, group_ids, residual
        )
        reg = ridge * D.square().sum()
        dual = lam @ errors + reg
        trace.append(dict(call=len(trace)+1, primal=float(errors.max()+reg), dual=float(dual)))
        return D, errors, reg, dual

    uniform = np.ones(groups) / groups
    tangent = np.zeros(groups)
    tangent[:2] = [1 / np.sqrt(2), -1 / np.sqrt(2)]
    _, error, _, _ = solve(uniform)
    ad = float(error.cpu().numpy() @ tangent)
    step = 1e-5
    fd = float((solve(uniform+step*tangent)[3]-solve(uniform-step*tangent)[3])/(2*step))
    assert abs(fd-ad) < 1e-6*(1+abs(ad)), (fd, ad)

    def objective(weights):
        _, errors, _, value = solve(weights)
        return -float(value), -errors.cpu().numpy()

    opt = minimize(objective, uniform, jac=True, bounds=[(0.,1.)]*groups,
                   constraints=[dict(type='eq',fun=lambda p:p.sum()-1,
                                     jac=lambda p:np.ones(groups))],
                   method='SLSQP', options=dict(maxiter=200,ftol=1e-8))
    weights = np.maximum(opt.x, 0)
    weights /= weights.sum()
    D, errors, reg, dual = solve(weights)
    zero = projected.square().sum(0)
    zero[:n] += free.square().sum(0)
    E0 = float(torch.zeros(groups,dtype=K.dtype,device=K.device).scatter_add_(0,group_ids,zero).max())
    assert float(D.norm()/B.norm()) <= np.sqrt(E0/rho)*(1+1e-7)
    return D, dict(errors=errors.cpu(), weights=weights, trace=trace,
                   success=bool(opt.success), message=str(opt.message), iterations=int(opt.nit),
                   relative_dual_gap=float((errors.max()+reg-dual)/(1+errors.max()+reg)),
                   envelope_derivative_error=abs(fd-ad), B_relative_bound=float(np.sqrt(E0/rho)))
