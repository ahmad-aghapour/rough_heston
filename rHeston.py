"""
rough_vol.py

Refactored utilities for:
  - Sum-of-exponentials approximation of (t-s)^(-alpha)
  - GPU-accelerated rough-volatility Monte Carlo (PyTorch)
  - Simple "implied vol surface" placeholder (returns option prices as in original)

Functionality preserved from the original script.
"""

from __future__ import annotations

from typing import Callable, Tuple

import math
import numpy as np
import torch


__all__ = [
    "compute_truncation_bound",
    "build_dyadic_intervals",
    "gauss_legendre_nodes_weights",
    "build_exponential_sum",
    "f_heston_like_torch",
    "g_heston_like_torch",
    "simulate_rough_vol_batch_torch",
    "compute_vol_surface",
]


# ---------------------------------------------------------------------------
#  Sum-of-exponentials kernel approximation helpers
# ---------------------------------------------------------------------------

def compute_truncation_bound(Delta_t: float, epsilon: float, alpha: float) -> float:
    """
    Compute the truncation bound L used in the dyadic interval construction.

    This follows the original formula and guards against non-positive 'tmp'.
    """
    tmp = -math.log(epsilon / 3.0) - (1.0 - alpha) * math.log(Delta_t)
    if tmp <= 0:
        tmp = 1.0
    # L = (1/Delta_t)^(1-alpha) * tmp^(1-alpha)
    L = (Delta_t ** (alpha - 1.0)) * (tmp ** (1.0 - alpha))
    return L


def build_dyadic_intervals(
    Delta_t: float, epsilon: float, alpha: float
) -> Tuple[int, int, float]:
    """
    Build dyadic interval indices [j_min, j_max] and return L.
    Behavior is preserved (including the fallback j_min = -10).
    """
    gamma_part = math.gamma(1.0 - alpha)
    const_val = gamma_part * (1.0 - alpha) * (epsilon / 3.0)
    j_min = math.ceil(math.log(const_val, 2)) if const_val > 0 else -10

    L = compute_truncation_bound(Delta_t, epsilon, alpha)
    j_max = math.ceil(math.log(L, 2)) - 1 if L > 1e-15 else j_min
    return j_min, j_max, L


def gauss_legendre_nodes_weights(M: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss–Legendre nodes and weights on [-1, 1].
    """
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(M)
    return x, w


def build_exponential_sum(
    alpha: float, Delta_t: float, epsilon: float, M: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the sum-of-exponentials approximation for the fractional kernel.

    Returns:
        lambdas: shape (K,) array of exponents
        omegas:  shape (K,) array of weights
    """
    gamma_ = 1.0 / (1.0 - alpha)
    prefactor = 1.0 / (math.gamma(1.0 - alpha) * (1.0 - alpha))
    j_min, j_max, L = build_dyadic_intervals(Delta_t, epsilon, alpha)

    lambdas: list[float] = []
    omegas: list[float] = []

    x_gl, w_gl = gauss_legendre_nodes_weights(M)

    for j in range(j_min, j_max + 1):
        a_j = 2.0**j
        b_j = min(2.0 ** (j + 1), L)
        if a_j >= L:
            break

        half_width = 0.5 * (b_j - a_j)
        midpoint = 0.5 * (b_j + a_j)

        for m in range(M):
            eta_jm = half_width * x_gl[m] + midpoint
            W_jm = half_width * w_gl[m]
            lam_k = eta_jm ** gamma_
            omg_k = prefactor * W_jm
            lambdas.append(lam_k)
            omegas.append(omg_k)

        if b_j >= L:
            break

    return np.asarray(lambdas, dtype=float), np.asarray(omegas, dtype=float)


# ---------------------------------------------------------------------------
#  Example drift/diffusion specs (Heston-like)
# ---------------------------------------------------------------------------

def f_heston_like_torch(V: torch.Tensor, kappa: float = 0.1, theta: float = 0.13156) -> torch.Tensor:
    """
    Vectorized drift: f(V) = kappa * (theta - V)
    """
    return kappa * (theta - V)


def g_heston_like_torch(V: torch.Tensor, nu: float = 0.131) -> torch.Tensor:
    """
    Vectorized diffusion: g(V) = nu * sqrt(max(V, 0))
    """
    V_clipped = torch.clamp(V, min=0.0)
    return nu * torch.sqrt(V_clipped)


# ---------------------------------------------------------------------------
#  Rough-volatility simulator (GPU-capable)
# ---------------------------------------------------------------------------

def simulate_rough_vol_batch_torch(
    alpha: float,                        # 0 < alpha < 0.5
    f: Callable[[torch.Tensor], torch.Tensor],
    g: Callable[[torch.Tensor], torch.Tensor],
    V0: float = 0.04,
    S0: float = 100.0,
    r: float = 0.0,
    rho: float = 0.0,
    T: float = 1.0,
    N: int = 100,
    N_exp: int = 3,
    M: int = 10_000,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate M paths of a rough volatility model in parallel with PyTorch + CUDA.

        dS_t = r S_t dt + S_t sqrt(V_t) dW_t
        V_t  = V_0 + [kernel convolution of f(V_s) ds + g(V_s) dB_s] / Gamma(1-alpha)

    Uses a sum-of-exponentials approximation for (t-s)^{-alpha}.

    Returns:
        t_vals: shape (N+1,)
        S_vals: shape (M, N+1)
        V_vals: shape (M, N+1)
    """
    # 0) Device selection
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Time grid
    tau = T / N
    t_vals = torch.linspace(0.0, T, steps=N + 1, dtype=torch.float64, device=device)

    # 2) Kernel approx (note: original script uses 1 - alpha here)
    x_array_np, w_array_np = build_exponential_sum(1.0 - alpha, 1e-5, 1e-4, M=N_exp)
    x_array = torch.tensor(x_array_np, device=device, dtype=torch.float64)
    w_array = torch.tensor(w_array_np, device=device, dtype=torch.float64)

    # 3) Path containers
    S_vals = torch.zeros((M, N + 1), device=device, dtype=torch.float64)
    V_vals = torch.zeros((M, N + 1), device=device, dtype=torch.float64)
    S_vals[:, 0] = S0
    V_vals[:, 0] = V0

    # Memory terms H, J: shape (M, N_exp)
    H = torch.zeros((M, x_array.shape[0]), device=device, dtype=torch.float64)
    J = torch.zeros((M, x_array.shape[0]), device=device, dtype=torch.float64)

    # Gamma constants (CPU float fine, but we keep dtype for clarity)
    gamma_1_minus_alpha = math.gamma(1.0 - alpha)
    gamma_2_minus_alpha = math.gamma(2.0 - alpha)

    # 4) Correlated increments covariance
    c11 = tau ** (1.0 - 2.0 * alpha) / (1.0 - 2.0 * alpha) if alpha != 0.5 else 0.0
    c12 = tau ** (1.0 - alpha) / (1.0 - alpha)
    c13 = rho * tau ** (1.0 - alpha) / (1.0 - alpha)
    c22 = tau
    c23 = rho * tau
    c33 = tau

    cov_matrix = torch.tensor(
        [[c11, c12, c13],
         [c12, c22, c23],
         [c13, c23, c33]],
        device=device, dtype=torch.float64
    )

    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(3, device=device, dtype=torch.float64),
        covariance_matrix=cov_matrix
    )

    # Precompute exp(-x_l * tau)
    e_term_1d = torch.exp(-x_array * tau)  # shape (N_exp,)

    # 5) Time stepping
    for n in range(1, N + 1):
        # (a) Correlated draws: [Z^V, Z^J, Z^stock], shape (M, 3)
        random_vector = mvn.sample((M,))  # (M, 3)
        Z_V = random_vector[:, 0]
        Z_J = random_vector[:, 1]
        Z_stock = random_vector[:, 2]

        # (b) Previous values
        V_prev = V_vals[:, n - 1]
        S_prev = S_vals[:, n - 1]

        # (c) Evaluate f, g
        f_val = f(V_prev)  # (M,)
        g_val = g(V_prev)  # (M,)

        # (d) Rough-vol update for V (formula preserved)
        term_drift = (tau ** (1.0 - alpha) / gamma_2_minus_alpha) * f_val
        term_hist = (1.0 / gamma_1_minus_alpha) * torch.sum(
            w_array * e_term_1d * H, dim=1
        )
        term_diff = (1.0 / gamma_1_minus_alpha) * g_val * Z_V
        term_histJ = (1.0 / gamma_1_minus_alpha) * torch.sum(
            w_array * e_term_1d * J, dim=1
        )

        V_new = V0 + term_drift + term_hist + term_diff + term_histJ
        V_new = torch.clamp(V_new, min=0.0)
        V_vals[:, n] = V_new

        # (e) Update memory terms H and J
        f_val_2d = f_val.unsqueeze(1)  # (M,1)
        numerator = 1.0 - e_term_1d    # (N_exp,)
        small_x_mask = (x_array < 1.0e-5)

        incr_H_small = f_val_2d * tau
        incr_H_large = f_val_2d * (numerator / x_array)
        incr_H = torch.where(small_x_mask.unsqueeze(0), incr_H_small, incr_H_large)
        H = incr_H + e_term_1d.unsqueeze(0) * H

        g_val_2d = g_val.unsqueeze(1)   # (M,1)
        Z_J_2d = Z_J.unsqueeze(1)       # (M,1)
        incr_J = g_val_2d * Z_J_2d * e_term_1d.unsqueeze(0)
        J = incr_J + e_term_1d.unsqueeze(0) * J

        # (f) Log-Euler update for S with V_prev
        X_prev = torch.log(S_prev)
        drift_log = (r - 0.5 * V_prev) * tau
        diffusion_log = torch.sqrt(V_prev) * Z_stock
        X_new = X_prev + drift_log + diffusion_log
        S_vals[:, n] = torch.exp(X_new)

    return t_vals, S_vals, V_vals


# ---------------------------------------------------------------------------
#  "Vol surface" computation (returns option prices on a grid, as per original)
# ---------------------------------------------------------------------------

def compute_vol_surface(
    params: tuple[float, float, float, float, float, float],
    device: torch.device = torch.device("cuda"),
    show: bool = False
) -> np.ndarray:
    """
    Build a grid of call prices (kept as 'implied_vol_surface' in the original code).

    Args:
        params: (alpha, theta, rho, V0, kappa, nu)
        device: torch device to run the simulation on (defaults to 'cuda')
        show:   if True, prints the price per (T, K) as in the original

    Returns:
        np.ndarray of shape (len(T_range), len(K_range)) with call prices
    """
    alpha, theta, rho, V0, kappa, nu = params

    T = 1.0
    N = 500
    N_exp = 2

    S0 = 100.0
    r = 0.0

    # Original constants (strike & MC settings) preserved
    K = 120.0  # unused downstream but kept for parity with original variables
    M = 100_000

    # Local drift/diffusion wrappers binding parameters
    def f_heston_like_torch_mod(V: torch.Tensor, kappa_: float = kappa, theta_: float = theta) -> torch.Tensor:
        return kappa_ * (theta_ - V)

    def g_heston_like_torch_mod(V: torch.Tensor, nu_: float = nu) -> torch.Tensor:
        V_clipped = torch.clamp(V, min=0.0)
        return nu_ * torch.sqrt(V_clipped)

    # Run simulation
    t_vals, S_vals, _ = simulate_rough_vol_batch_torch(
        alpha=alpha,
        f=f_heston_like_torch_mod,
        g=g_heston_like_torch_mod,
        V0=V0,
        S0=S0,
        r=r,
        rho=rho,
        T=T,
        N=N,
        N_exp=N_exp,
        M=M,
        device=device,
    )

    # --- Small, non-invasive fix: use NumPy arrays when indexing with NumPy ops
    t_vals_np = t_vals.detach().cpu().numpy()
    S_vals_np = S_vals.detach().cpu().numpy()

    # Grids (kept identical in spirit to the original)
    T_range = np.linspace(0.1, 1.0, num=32)
    K_range = np.linspace(90.0, 110.0, num=32)

    # Placeholder surface (actually call prices, as in original code)
    surface = np.zeros((len(T_range), len(K_range)))

    for i, T_i in enumerate(T_range):
        # Index closest simulated time to T_i
        n_idx = int(np.argmin(np.abs(t_vals_np - T_i)))
        S_T = S_vals_np[:, n_idx]  # shape (M,)

        discount_factor = math.exp(-r * T_i)
        for j, K_j in enumerate(K_range):
            payoffs_j = np.fmax(S_T - K_j, 0.0)
            option_price_j = discount_factor * np.mean(payoffs_j)

            if show:
                print(
                    f"Rough-Heston call option price with strike K={K_j:.2f}, "
                    f"T={T_i}, M={S_vals_np.shape[0]} paths, N={N} steps:"
                )
                print(f"Monte Carlo Estimate ≈ {option_price_j:.4f}")

            # The original sets 'implied vol' equal to the price. Preserved here.
            surface[i, j] = option_price_j

    return surface


# ---------------------------------------------------------------------------
# Optional: quick self-test (kept minimal; not executed on import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny smoke test to ensure functions run (reduced sizes for speed)
    alpha = 0.3
    params = (alpha, 0.13156, -0.5, 0.04, 0.1, 0.131)

    # Use CPU by default for a light test
    dev = torch.device("cpu")

    t_vals, S_vals, V_vals = simulate_rough_vol_batch_torch(
        alpha=alpha,
        f=lambda V: f_heston_like_torch(V, kappa=0.1, theta=0.13156),
        g=lambda V: g_heston_like_torch(V, nu=0.131),
        V0=0.04,
        S0=100.0,
        r=0.0,
        rho=-0.5,
        T=0.2,
        N=50,
        N_exp=2,
        M=500,
        device=dev,
    )
    surf = compute_vol_surface(params, device=dev, show=False)
    print("Shapes:", t_vals.shape, S_vals.shape, V_vals.shape, surf.shape)
