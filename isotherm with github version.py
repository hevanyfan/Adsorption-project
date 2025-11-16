from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

# ---------------------------
# Config
# ---------------------------
@dataclass
class BreakthroughConfig:
    # Table 2 (brief)
    L: float = 0.32          # m
    omega: float = 0.32      # bed void fraction
    rho_s: float = 990.0     # kg/m^3 (particle density)
    r_p: float = 1e-3        # m
    T: float = 303.0         # K
    P_tot: float = 100e3     # Pa
    R: float = 8.314         # m^3 Pa / (mol K)
    u_in: float = 4.864e-3   # m/s (superficial velocity)
    D_m: float = 7.48e-5     # m^2/s  (molecular diffusivity) e-5

    # LDF kinetics (tunable for sensitivity)
    k_N2: float = 0.1        # s^-1 0.1
    k_O2: float = 0.1        # s^-1 0.1

    # Discretization & solver
    Nz: int = 80
    t_end: float = 20.0    # s
    atol: float = 1e-6
    rtol: float = 1e-4
    use_interstitial_velocity: bool = False  # True → use u_in/omega

    # Output
    outdir: str = "outputs"

CFG = BreakthroughConfig()
os.makedirs(CFG.outdir, exist_ok=True)

# ---------------------------
# Table 1 data (Romulus/Remus × N2/O2)
# ---------------------------
P_N2_Romulus = np.array([0.0, 2026.5, 3039.75, 4053.0, 6079.5, 8106.0, 61909.575, 111356.175, 173265.75, 247435.65, 321706.875, 395978.1, 470148.0, 544419.225, 618690.45])
q_N2_Romulus = np.array([0.000, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835, 1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861])

P_O2_Romulus = np.array([0.0, 8106.0, 20265.0, 247435.65, 371153.475, 494871.3, 618690.45])
q_O2_Romulus = np.array([0.000, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815])

P_N2_Remus = np.array([0.0, 63400.0, 88700.0, 117500.0, 162500.0, 208500.0, 303800.0, 427000.0, 609500.0, 776600.0, 919000.0])
q_N2_Remus = np.array([0.000, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472])

P_O2_Remus = np.array([0.0, 33200.0, 61300.0, 96100.0, 143200.0, 214800.0, 355900.0, 548200.0, 747900.0, 859000.0, 931700.0])
q_O2_Remus = np.array([0.000, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348])

# ---------------------------
# Sips isotherm + fitting
# ---------------------------
def sips_isotherm(P, q_max, b, n):
    """Sips isotherm q(P) with P in Pa, q in mol/kg."""
    return q_max * (b * P)**n / (1 + (b * P)**n)

def fit_sips(P, q):
    initial_guesses = [2.0, 1e-5, 0.9]
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
    params, cov = curve_fit(
        sips_isotherm, P, q, p0=initial_guesses, bounds=bounds, maxfev=200000
    )
    return params, cov

def get_all_sips_params():
    pars = {}
    for key, (P, q) in {
        ("Romulus","N2"): (P_N2_Romulus, q_N2_Romulus),
        ("Romulus","O2"): (P_O2_Romulus, q_O2_Romulus),
        ("Remus","N2"):   (P_N2_Remus,   q_N2_Remus),
        ("Remus","O2"):   (P_O2_Remus,   q_O2_Remus),
    }.items():
        (qmax, b, n), cov = fit_sips(P, q)
        pars[key] = dict(qmax=float(qmax), b=float(b), n=float(n))
    return pars

def sips_q(pars, material, gas, P_partial):
    p = pars[(material, gas)]
    return sips_isotherm(P_partial, p['qmax'], p['b'], p['n'])

# ---------------------------
# Breakthrough model
# ---------------------------
def axial_dispersion(u_in, r_p, D_m):
    """DL = 0.7*D_m + u_in*r_p  (per brief)."""
    return 0.7*D_m + u_in*r_p

def rhs(t, y, *, pars, cfg):
    """
    y stacks: [yN2 (Nz), yO2 (Nz), qN2 (Nz), qO2 (Nz)]
    PDE for yi with axial dispersion & convection + LDF source term from solid phase.
    """
    Nz = cfg.Nz
    dz = cfg.L/(Nz-1)
    u = cfg.u_in if not cfg.use_interstitial_velocity else cfg.u_in/cfg.omega
    DL = axial_dispersion(cfg.u_in, cfg.r_p, cfg.D_m)

    # unpack
    yN2 = y[0:Nz]
    yO2 = y[Nz:2*Nz]
    qN2 = y[2*Nz:3*Nz]
    qO2 = y[3*Nz:4*Nz]

    # renormalize (avoid drift)
    s = yN2 + yO2 + 1e-16
    yN2 = np.clip(yN2/s, 0.0, 1.0)
    yO2 = np.clip(yO2/s, 0.0, 1.0)

    # equilibrium loadings (local partial pressure)
    qN2_star = sips_q(pars, cfg.material, "N2", cfg.P_tot*yN2)
    qO2_star = sips_q(pars, cfg.material, "O2", cfg.P_tot*yO2)

    # LDF kinetics
    dqN2_dt = cfg.k_N2 * (qN2_star - qN2)
    dqO2_dt = cfg.k_O2 * (qO2_star - qO2)

    # Danckwerts inlet & Neumann outlet
    alpha = u*dz/DL if DL > 0 else 0.0
    def inlet_value(y1, yfeed):
        return (y1 + alpha*yfeed)/(1.0+alpha) if DL > 0 else yfeed

    yN2_0  = inlet_value(yN2[1], 1.0)  # feed = pure N2
    yO2_0  = inlet_value(yO2[1], 0.0)
    yN2_Np1 = yN2[-1]
    yO2_Np1 = yO2[-1]

    # derivatives
    def second_derivative(yf, y0, yNp1):
        ym1 = np.empty_like(yf); yp1 = np.empty_like(yf)
        ym1[0] = y0;      ym1[1:] = yf[:-1]
        yp1[:-1] = yf[1:]; yp1[-1] = yNp1
        return (yp1 - 2.0*yf + ym1)/dz**2

    def first_derivative_upwind(yf, y0):
        ym1 = np.empty_like(yf); ym1[0] = y0; ym1[1:] = yf[:-1]
        return (yf - ym1)/dz

    d2yN2 = second_derivative(yN2, yN2_0, yN2_Np1)
    d2yO2 = second_derivative(yO2, yO2_0, yO2_Np1)
    dyN2  = first_derivative_upwind(yN2, yN2_0)
    dyO2  = first_derivative_upwind(yO2, yO2_0)

    factor = (cfg.R*cfg.T/cfg.P_tot) * (cfg.rho_s*(1.0-cfg.omega)/cfg.omega)
    dyN2_dt = DL*d2yN2 - u*dyN2 + factor*dqN2_dt
    dyO2_dt = DL*d2yO2 - u*dyO2 + factor*dqO2_dt

    return np.concatenate([dyN2_dt, dyO2_dt, dqN2_dt, dqO2_dt])


def rhs_two_gas(t, Y, cfg: BreakthroughConfig, sips_params_mat):
    """
    Two-gas (N2 + O2) breakthrough model.
    State vector Y of length 4N:

      Y = [yN2(0..N-1), yO2(0..N-1), qN2(0..N-1), qO2(0..N-1)]
    """
    N = cfg.N
    dz = cfg.L / (N - 1)

    # velocity (superficial)
    v = cfg.u_in

    # axial dispersion (set to 0 if you want convection-only)
    D_L = 0.0  # or: 0.7 * cfg.D_m + cfg.u_in * cfg.r_p

    # ---- unpack state vector ----
    yN2 = Y[0:N]
    yO2 = Y[N:2*N]
    qN2 = Y[2*N:3*N]
    qO2 = Y[3*N:4*N]

    # ---- renormalise gas mole fractions (avoid drift) ----
    s = yN2 + yO2 + 1e-16
    yN2 = np.clip(yN2 / s, 0.0, 1.0)
    yO2 = np.clip(yO2 / s, 0.0, 1.0)

    # ---- Sips equilibrium loadings ----
    qmax_N2, b_N2, n_N2 = sips_params_mat["N2"]
    qmax_O2, b_O2, n_O2 = sips_params_mat["O2"]

    P_N2 = cfg.P_tot * yN2
    P_O2 = cfg.P_tot * yO2

    qN2_star = sips_isotherm(P_N2, qmax_N2, b_N2, n_N2)
    qO2_star = sips_isotherm(P_O2, qmax_O2, b_O2, n_O2)

    # ---- LDF kinetics (solid phase) ----
    dqN2_dt = cfg.k_N2 * (qN2_star - qN2)
    dqO2_dt = cfg.k_O2 * (qO2_star - qO2)

    # ==============================
    # 1) CONVECTION (GitHub style)
    # ==============================
    # inlet: feed = pure N2 (yN2=1, yO2=0)
    yN2_feed, yO2_feed = 1.0, 0.0

    # build "convection-only" extended arrays (left ghost only)
    # [feed, y0, y1, ..., y_{N-1}]
    yN2_conv_ext = np.insert(yN2, 0, yN2_feed)
    yO2_conv_ext = np.insert(yO2, 0, yO2_feed)

    # GitHub-style upwind:
    # dc/dt_conv = v * (upstream - current) / dz
    dyN2_dt_conv = v * (yN2_conv_ext[:-1] - yN2_conv_ext[1:]) / dz
    dyO2_dt_conv = v * (yO2_conv_ext[:-1] - yO2_conv_ext[1:]) / dz

    # ==============================
    # 2) DISPERSION (needs both ghosts)
    # ==============================
    # outlet: zero-gradient (Neumann): y_N = y_{N-1}
    yN2_right = yN2[-1]
    yO2_right = yO2[-1]

    # extended arrays for diffusion: [ghost_left, interior..., ghost_right]
    # left ghost for diffusion = same as feed here
    yN2_ext = np.concatenate(([yN2_feed], yN2, [yN2_right]))
    yO2_ext = np.concatenate(([yO2_feed], yO2, [yO2_right]))

    # central second derivative: (y_{i+1} - 2y_i + y_{i-1}) / dz^2
    d2yN2_dz2 = (yN2_ext[2:] - 2.0*yN2_ext[1:-1] + yN2_ext[:-2]) / dz**2
    d2yO2_dz2 = (yO2_ext[2:] - 2.0*yO2_ext[1:-1] + yO2_ext[:-2]) / dz**2

    dyN2_dt_diff = D_L * d2yN2_dz2
    dyO2_dt_diff = D_L * d2yO2_dz2

    # ==============================
    # 3) ADSORPTION COUPLING
    # ==============================
    factor = (cfg.R * cfg.T / cfg.P_tot) * (cfg.rho_s * (1.0 - cfg.eps) / cfg.eps)
    dyN2_dt_ads = factor * dqN2_dt
    dyO2_dt_ads = factor * dqO2_dt

    # total gas-phase derivatives
    dyN2_dt = dyN2_dt_conv + dyN2_dt_diff + dyN2_dt_ads
    dyO2_dt = dyO2_dt_conv + dyO2_dt_diff + dyO2_dt_ads

    # ---- pack back into single vector ----
    return np.concatenate([dyN2_dt, dyO2_dt, dqN2_dt, dqO2_dt])
