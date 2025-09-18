"""
Augumented Lagrangian Method for Low-Rank Euclidean Distance Matrix Completion
"Exact Reconstruction of Euclidean Distance Geometry Problem Using Low-rank Matrix Completion" by Abiy Tasissa, Rongjie Lai et al.
Original Code in MATLAB : https://github.com/abiy-tasissa/Nonconvex-Euclidean-Distance-Geometry-Problem-Solver
"""
import numpy as np
import torch
from math_utils import gradient, A_operator, At_operator


def BBGradient(x, fun, opts):
    """
    Barzilai-Borwein gradient method for optimization.
    
    Args:
        x: Initial point (torch tensor)
        fun: Function that returns (f, g) where f is objective and g is gradient
        opts: Dictionary with optimization options
    
    Returns:
        xmin: Optimal point found
    """
    device = x.device
    f, g = fun(x)
    gnorm = torch.norm(g)

    Q = 1.0
    C = f.item()
    alpha = opts['alpha']
    xmin = x.clone()
    fmin = f.item()

    n = x.size(0)
    MAXalpha = 1e16
    MINalpha = 1e-16

    for it in range(opts['maxit']):
        xold = x.clone()
        fold = f.item()
        gold = g.clone()

        numlinesearch = 1
        wolfe_factor = opts['rho'] * gnorm**2

        while True:
            x = xold - alpha * gold
            f_new, g_new = fun(x)

            if f_new <= C - alpha * wolfe_factor or numlinesearch >= 5:
                break
            alpha *= opts['sigma']
            numlinesearch += 1

        if f_new < fmin:
            xmin = x.clone()
            fmin = f_new.item()

        g = g_new
        gnorm = torch.norm(g)
        s = x - xold
        xstop = torch.norm(s) / np.sqrt(n)
        fstop = abs(fold - f_new.item()) / (abs(fold) + 1)

        if (xstop < opts['xtol'] and fstop < opts['ftol']) or (gnorm < opts['gtol']):
            break

        y = g - gold
        sy = abs(torch.trace(s.t() @ y))
        if sy == 0:
            break
        if it % 2 == 0:
            alpha = torch.norm(s)**2 / sy
        else:
            alpha = sy / torch.norm(y)**2

        alpha = torch.clamp(alpha, MINalpha, MAXalpha).item()

        Qold = Q
        Q = opts['eta'] * Qold + 1
        C = (opts['eta'] * Qold * C + f_new.item()) / Q

    return xmin


def alternating_completion(Dist, Weight, opts, lsopts):
    """    
    Args:
        Dist: Distance matrix (observed entries)
        Weight: Binary mask indicating observed entries
        opts: Dictionary with completion options
        lsopts: Dictionary with line search options
    
    Returns:
        P: Factor matrix (numpy array)
    """
    Dist_np = np.array(Dist, dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dist_t = torch.tensor(Dist_np, dtype=torch.float32, device=device)
    Weight_np = np.array(Weight, dtype=float)
    Weight_t = torch.tensor(Weight_np, dtype=torch.float32, device=device)
    
    r = opts['r']
    Rk = opts['rank']
    dim = Dist_t.size(0)

    I, J = torch.where(Weight_t == 1)
    edgeind = I * dim + J
    M = Dist_t.view(-1)[edgeind]
    b = M.clone()

    P = torch.rand(dim, Rk, device=device)
    D1 = torch.zeros_like(b, device=device)

    for i in range(opts['maxit']):
        P = BBGradient(P, lambda x: gradient(x, edgeind, I, J, dim, b, D1, r), lsopts)
        tmperr = A_operator(P, edgeind, I, J, dim) - b
        D1 += tmperr

        # Simple convergence check
        E1_current = 0.5 * r * torch.norm(tmperr)**2
        if E1_current < opts['tol']:
            break

    return P.cpu().numpy()
