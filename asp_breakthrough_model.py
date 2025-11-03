# Advanced Separation Processes - Part 2: Breakthrough Simulation
# Author: <your name>
# Course: CHEN40461/60461 - Advanced Separation Processes AY 2025-26
# Run in Jupyter:
#     %run asp_breakthrough_model.py
# Or in terminal:
#     python asp_breakthrough_model.py

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
    D_m: float = 7.48e-5     # m^2/s  (molecular diffusivity)

    # LDF kinetics (tunable for sensitivity)
    k_N2: float = 0.1        # s^-1
    k_O2: float = 0.1        # s^-1

    # Discretization & solver
    Nz: int = 80
    t_end: float = 1200.0    # s
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

def simulate(pars, cfg, material):
    Nz = cfg.Nz
    # initial: bed & solid saturated with O2
    yN2_0 = np.zeros(Nz); yO2_0 = np.ones(Nz)
    qN2_0 = np.zeros(Nz); qO2_0 = sips_q(pars, material, "O2", cfg.P_tot*np.ones(Nz))
    y0 = np.concatenate([yN2_0, yO2_0, qN2_0, qO2_0])

    # events on outlet yN2
    def ev(thr):
        def f(t,y): return y[Nz-1] - thr
        f.terminal = False; f.direction = 1
        return f

    cfg_local = BreakthroughConfig(**asdict(CFG))
    cfg_local.material = material  # runtime attach

    sol = solve_ivp(lambda t,Y: rhs(t,Y, pars=pars, cfg=cfg_local),
                    (0.0, cfg.t_end), y0, method="BDF",
                    atol=cfg.atol, rtol=cfg.rtol,
                    events=[ev(0.05), ev(0.50), ev(0.95)])

    t = sol.t
    yN2_out = sol.y[0:Nz, :][-1, :]
    yO2_out = sol.y[Nz:2*Nz, :][-1, :]
    t5  = sol.t_events[0][0] if len(sol.t_events[0])>0 else np.nan
    t50 = sol.t_events[1][0] if len(sol.t_events[1])>0 else np.nan
    t95 = sol.t_events[2][0] if len(sol.t_events[2])>0 else np.nan

    return dict(t=t, yN2_out=yN2_out, yO2_out=yO2_out,
                t5=t5, t50=t50, t95=t95, success=sol.success, message=sol.message)

def run_baseline_and_export():
    # 1) fit
    pars = get_all_sips_params()

    # 2) simulate
    rom = simulate(pars, CFG, "Romulus")
    rem = simulate(pars, CFG, "Remus")

    # 3) export breakthrough times
    bt = pd.DataFrame([
        {"Material":"Romulus","t_5% (s)":rom["t5"],"t_50% (s)":rom["t50"],"t_95% (s)":rom["t95"],"Success":rom["success"]},
        {"Material":"Remus","t_5% (s)":rem["t5"],"t_50% (s)":rem["t50"],"t_95% (s)":rem["t95"],"Success":rem["success"]},
    ])
    bt_path = os.path.join(CFG.outdir, "breakthrough_times.csv")
    bt.to_csv(bt_path, index=False)

    # 4) export curves on common time grid
    tmax = float(max(rom["t"][-1], rem["t"][-1]))
    t_common = np.linspace(0.0, tmax, 400)
    def interp(t_src, y_src, t_new): return np.interp(t_new, t_src, y_src)
    curves = pd.DataFrame({
        "time_s": t_common,
        "yN2_out_Romulus": interp(rom["t"], rom["yN2_out"], t_common),
        "yN2_out_Remus":   interp(rem["t"], rem["yN2_out"], t_common),
        "yO2_out_Romulus": interp(rom["t"], rom["yO2_out"], t_common),
        "yO2_out_Remus":   interp(rem["t"], rem["yO2_out"], t_common),
    })
    csv_path = os.path.join(CFG.outdir, "breakthrough_curves.csv")
    curves.to_csv(csv_path, index=False)

    # 5) plot and save
    plt.figure(figsize=(6,4))
    plt.plot(rom["t"], rom["yN2_out"], label="Romulus")
    plt.plot(rem["t"], rem["yN2_out"], label="Remus")
    plt.xlabel("Time (s)"); plt.ylabel("Outlet $y_{N2}$ (-)")
    plt.title("N2 Breakthrough at Outlet\n(feed: N2=1.0; bed initially O2-saturated)")
    plt.legend(); plt.tight_layout()
    fig_path = os.path.join(CFG.outdir, "breakthrough_yN2.png")
    plt.savefig(fig_path, dpi=200); plt.close()

    print("[OK] Baseline finished.")
    print(f"  → Breakthrough times: {bt_path}")
    print(f"  → Curves CSV:         {csv_path}")
    print(f"  → Plot PNG:           {fig_path}")

def sensitivity_scan(velocities=(2e-3, 4.864e-3, 8e-3), lengths=(0.2, 0.32, 0.5)):
    """Grid scan on u_in and L; writes outputs/sensitivity_times.csv"""
    pars = get_all_sips_params()
    rows = []
    for u in velocities:
        for L in lengths:
            cfg = BreakthroughConfig(**asdict(CFG))
            cfg.u_in = u
            cfg.L = L
            for mat in ("Romulus","Remus"):
                res = simulate(pars, cfg, mat)
                rows.append({
                    "u_in (m/s)": u, "L (m)": L, "Material": mat,
                    "t_5% (s)": res["t5"], "t_50% (s)": res["t50"], "t_95% (s)": res["t95"],
                    "Success": res["success"]
                })
    out = os.path.join(CFG.outdir, "sensitivity_times.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] Sensitivity exported: {out}")

if __name__ == "__main__":
    run_baseline_and_export()
    # Optional:
    # sensitivity_scan()
