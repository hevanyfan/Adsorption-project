# Advanced Separation Processes - Part 2: Breakthrough Simulation
# Author: 
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
    D_m: float = 7.48e-5     # m^2/s  (molecular diffusivity) e-5
    tau: float = 3.0 

    # LDF kinetics (tunable for sensitivity)
    k_N2: float = None        # s^-1 0.1
    k_O2: float = None       # s^-1 0.1

    # Discretization & solver
    Nz: int = 80
    t_end: float = 0.02    # s
    atol: float = 1e-6
    rtol: float = 1e-4
    use_interstitial_velocity: bool = False  # True → use u_in/omega

    # Output
    outdir: str = "outputs"

CFG = BreakthroughConfig()
os.makedirs(CFG.outdir, exist_ok=True)

def k_glueckauf_from_Dm(cfg: BreakthroughConfig):
    # Dp ≈ Dm / tau
    Dp = cfg.D_m / cfg.tau
    return 15.0 * Dp / (cfg.r_p**2)
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

def rhs(t, Y, *, pars, cfg):
    """
    Two-gas (N2 + O2) breakthrough model.

    GitHub-style:
      - 用 np.insert 做左边 ghost（对流）
      - 用 np.concatenate 做左右 ghost（二阶导）
      - 全部向量化，无 for 循环
    """
    Nz = cfg.Nz
    dz = cfg.L / (Nz - 1)
    # 速度：可以选用 interstitial 或 superficial
    u = cfg.u_in / cfg.omega if cfg.use_interstitial_velocity else cfg.u_in

    # 轴向弥散 (brief): D_L = 0.7*D_m + u_in*r_p
    D_L = 0.7 * cfg.D_m + cfg.u_in * cfg.r_p

    # ---------- unpack 状态向量 ----------
    yN2 = Y[0:Nz]
    yO2 = Y[Nz:2*Nz]
    qN2 = Y[2*Nz:3*Nz]
    qO2 = Y[3*Nz:4*Nz]

    # ---------- 气相 mole fraction 归一化 ----------
    s = yN2 + yO2 + 1e-16
    yN2 = np.clip(yN2 / s, 0.0, 1.0)
    yO2 = np.clip(yO2 / s, 0.0, 1.0)

    # ---------- Sips 平衡吸附量（用原来的 sips_q + cfg.material） ----------
    qN2_star = sips_q(pars, cfg.material, "N2", cfg.P_tot * yN2)
    qO2_star = sips_q(pars, cfg.material, "O2", cfg.P_tot * yO2)

    # ---------- LDF 固相动力学 ----------
    dqN2_dt = cfg.k_N2 * (qN2_star - qN2)
    dqO2_dt = cfg.k_O2 * (qO2_star - qO2)

    # ===========================
    # (1) GitHub-style 对流项
    # ===========================
    # 入口：纯 N2
    yN2_feed, yO2_feed = 1.0, 0.0

    # 左边插一个 ghost cell（完全照 GitHub 写法）:
    #   [feed, y0, y1, ..., y_{Nz-1}]
    yN2_conv_ext = np.insert(yN2, 0, yN2_feed)
    yO2_conv_ext = np.insert(yO2, 0, yO2_feed)

    # GitHub 示例里的上风格式：
    # dy/dt_conv = u * (upstream - current) / dz
    dyN2_dt_conv = u * (yN2_conv_ext[:-1] - yN2_conv_ext[1:]) / dz
    dyO2_dt_conv = u * (yO2_conv_ext[:-1] - yO2_conv_ext[1:]) / dz

    # ===========================
    # (2) 二阶导扩散项（左右 ghost）
    # ===========================
    # 右边边界：零梯度 → ghost_right = 最后一个点
    yN2_right = yN2[-1]
    yO2_right = yO2[-1]

    # 对扩散来说，要左右 ghost 都有：
    #   [y_feed, y0, y1, ..., y_{Nz-1}, y_right]
    yN2_diff_ext = np.concatenate(([yN2_feed], yN2, [yN2_right]))
    yO2_diff_ext = np.concatenate(([yO2_feed], yO2, [yO2_right]))

    # 中心差分二阶导：
    # d2y/dz2 = (y_{i+1} - 2y_i + y_{i-1}) / dz^2
    d2yN2_dz2 = (yN2_diff_ext[2:] - 2.0 * yN2_diff_ext[1:-1] + yN2_diff_ext[:-2]) / dz**2
    d2yO2_dz2 = (yO2_diff_ext[2:] - 2.0 * yO2_diff_ext[1:-1] + yO2_diff_ext[:-2]) / dz**2

    dyN2_dt_diff = D_L * d2yN2_dz2
    dyO2_dt_diff = D_L * d2yO2_dz2

    # ===========================
    # (3) 吸附耦合项
    # ===========================
    factor = (cfg.R * cfg.T / cfg.P_tot) * (cfg.rho_s * (1.0 - cfg.omega) / cfg.omega)
    dyN2_dt_ads = factor * dqN2_dt
    dyO2_dt_ads = factor * dqO2_dt

    # 总的 dy/dt = 对流 + 扩散 + 吸附
    dyN2_dt = dyN2_dt_conv + dyN2_dt_diff + dyN2_dt_ads
    dyO2_dt = dyO2_dt_conv + dyO2_dt_diff + dyO2_dt_ads

    # ---------- 打包返回 ----------
    return np.concatenate([dyN2_dt, dyO2_dt, dqN2_dt, dqO2_dt])


def simulate(pars, cfg, material):
    Nz = cfg.Nz
    yN2_0 = np.zeros(Nz)
    yO2_0 = np.ones(Nz)
    qN2_0 = np.zeros(Nz)
    qO2_0 = sips_q(pars, material, "O2", cfg.P_tot * np.ones(Nz))
    y0 = np.concatenate([yN2_0, yO2_0, qN2_0, qO2_0])

    # --- Event function uses normalized outlet y_N2 ---
    def ev(thr):
        def f(t, Y):
            yN2 = Y[0:Nz]
            yO2 = Y[Nz:2*Nz]
            s = yN2 + yO2 + 1e-16
            yN2_out_norm = yN2[-1] / s[-1]
            return yN2_out_norm - thr
        f.terminal = False
        f.direction = 1
        return f

    cfg_local = BreakthroughConfig(**asdict(CFG))
    cfg_local.material = material

    sol = solve_ivp(lambda t, Y: rhs(t, Y, pars=pars, cfg=cfg_local),
                    (0.0, cfg.t_end), y0, method="BDF",
                    atol=cfg.atol, rtol=cfg.rtol,
                    events=[ev(0.05), ev(0.50), ev(0.95)])

    # --- Normalize outlet mole fractions before exporting ---
    yN2_raw = sol.y[0:Nz, :]
    yO2_raw = sol.y[Nz:2*Nz, :]
    s_out = yN2_raw[-1, :] + yO2_raw[-1, :] + 1e-16
    yN2_out = np.clip(yN2_raw[-1, :] / s_out, 0.0, 1.0)
    yO2_out = 1.0 - yN2_out  # binary system -> enforce consistency

    # Breakthrough times (already based on normalized y)
    t5  = sol.t_events[0][0] if len(sol.t_events[0]) > 0 else np.nan
    t50 = sol.t_events[1][0] if len(sol.t_events[1]) > 0 else np.nan
    t95 = sol.t_events[2][0] if len(sol.t_events[2]) > 0 else np.nan

    return dict(t=sol.t, yN2_out=yN2_out, yO2_out=yO2_out,
                t5=t5, t50=t50, t95=t95, success=sol.success)


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
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # --------- 上面：曲线 ---------
    ax1.plot(rom["t"], rom["yN2_out"], label="Romulus $y_{N2}$")
    ax1.plot(rem["t"], rem["yN2_out"], label="Remus $y_{N2}$")
    ax1.plot(rom["t"], rom["yO2_out"], "--", label="Romulus $y_{O2}$")
    ax1.plot(rem["t"], rem["yO2_out"], "--", label="Remus $y_{O2}$")

    ax1.set_ylabel("Outlet mole fraction (-)")
    ax1.set_title("Outlet breakthrough of N$_2$ and O$_2$")
    ax1.legend(loc="upper right")
    ax1.grid(False)

    # --------- 下面：表格 ---------
    ax2.axis("off")  # 不要坐标轴

    table_data = [
        [f"{rom['t5']:.3e}", f"{rom['t50']:.3e}", f"{rom['t95']:.3e}"],
        [f"{rem['t5']:.3e}", f"{rem['t50']:.3e}", f"{rem['t95']:.3e}"],
    ]
    col_labels = [r"$t_{5\%}$ (s)", r"$t_{50\%}$ (s)", r"$t_{95\%}$ (s)"]
    row_labels = ["Romulus", "Remus"]

    table = ax2.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)  # 调一点尺寸更好看

    ax2.set_xlabel("Time (s)")
    ax1.set_xlim(0.0, CFG.t_end)

    plt.tight_layout()
    plt.show()

    

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
    k_val = k_glueckauf_from_Dm(CFG)
    CFG.k_N2 = k_val
    CFG.k_O2 = k_val   # 如果你想 N2/O2 不同，也可以自己改成别的

    print(f"Using Glueckauf k = {k_val:.3e} s^-1")

    run_baseline_and_export()
    # Optional:
    # sensitivity_scan()
