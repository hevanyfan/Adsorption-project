# Advanced Separation Processes - Part 3: Equilibrium Theory Comparison
# Course: CHEN40461/60461 - Advanced Separation Processes AY 2025-26
# Run in Jupyter: %run asp_equilibrium_pde_only.py
# Or terminal:    python asp_equilibrium_pde_only.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CONFIG = {
    'L': 0.32,          # m
    'epsilon': 0.32,    # Void fraction
    'rho_s': 990.0,     # kg/m3
    'r_p': 1e-3,        # m
    'T': 303.0,         # K
    'P_tot': 100e3,     # Pa
    'u_in': 4.864e-3,   # m/s
    'D_m': 7.48e-5,     # m2/s
    'tau': 3.0,         # Tortuosity
    'R_gas': 8.314,
    
    'Nz': 50,           # Grid points
    't_end': 1500.0,    # End time
    'outdir': 'outputs_equilibrium'
}

os.makedirs(CONFIG['outdir'], exist_ok=True)

# ==============================================================================
# 2. DATA & ISOTHERMS
# ==============================================================================

Data_Rom_N2 = {'P': np.array([0, 2026.5, 3039.75, 4053.0, 6079.5, 8106.0, 61909.5, 111356.2, 173265.8, 247435.7, 321706.9, 395978.1, 470148.0, 544419.2, 618690.5]), 
               'q': np.array([0, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835, 1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861])}
Data_Rom_O2 = {'P': np.array([0, 8106.0, 20265.0, 247435.7, 371153.5, 494871.3, 618690.5]), 
               'q': np.array([0, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815])}

Data_Rem_N2 = {'P': np.array([0, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000]), 
               'q': np.array([0, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472])}
Data_Rem_O2 = {'P': np.array([0, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700]), 
               'q': np.array([0, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348])}

def sips_model(P, qmax, b, n):
    P_safe = np.maximum(P, 0.0)
    bP = b * P_safe
    try: term = bP**n
    except: term = np.full_like(P, 1e10)
    return qmax * term / (1.0 + term)

def fit_isotherm(data):
    p0 = [2.0, 1e-5, 0.9]
    bounds = ([0, 0, 0.1], [20, 1.0, 5.0])
    params, _ = curve_fit(sips_model, data['P'], data['q'], p0=p0, bounds=bounds, maxfev=10000)
    return params

# ==============================================================================
# 3. PDE ENGINE
# ==============================================================================

def calculate_physics(cfg, mode='real'):
    """
    Calculates parameters based on mode.
    mode='real': Uses Table 2 equations.
    mode='ideal': Sets DL=0 and k=Large to simulate Equilibrium Theory.
    """
    v_inter = cfg['u_in'] / cfg['epsilon']
    
    # Base calculations
    Dp = cfg['D_m'] / cfg['tau']
    k_real = 15.0 * Dp / (cfg['r_p']**2)
    DL_real = 0.7 * cfg['D_m'] + v_inter * cfg['r_p']
    
    if mode == 'ideal':
        # EQUILIBRIUM THEORY SETTINGS
        DL = 0.0            # No Dispersion
        k_LDF = k_real #* 1000.0 # Instant mass transfer (approx)
    else:
        # REAL SETTINGS
        DL = DL_real
        k_LDF = k_real

    term = (cfg['rho_s'] * (1.0 - cfg['epsilon']) / cfg['epsilon'])
    coupling = (cfg['R_gas'] * cfg['T'] / cfg['P_tot']) * term
    
    return v_inter, DL, k_LDF, coupling

def pde_rhs(t, y_flat, cfg, p_n2, p_o2, derived_params):
    v, DL, k, coupling = derived_params
    Nz = cfg['Nz']
    dz = cfg['L'] / (Nz - 1)
    
    yN2 = np.clip(y_flat[0:Nz], 0.0, 1.0)
    yO2 = np.clip(y_flat[Nz:2*Nz], 0.0, 1.0)
    qN2 = y_flat[2*Nz:3*Nz]
    qO2 = y_flat[3*Nz:4*Nz]
    
    qN2_star = sips_model(yN2 * cfg['P_tot'], *p_n2)
    qO2_star = sips_model(yO2 * cfg['P_tot'], *p_o2)
    dqN2_dt = k * (qN2_star - qN2)
    dqO2_dt = k * (qO2_star - qO2)
    
    # Derivatives
    d2yN2 = (yN2[2:] - 2*yN2[1:-1] + yN2[:-2]) / dz**2
    dyN2_dx = (yN2[1:-1] - yN2[:-2]) / dz
    d2yO2 = (yO2[2:] - 2*yO2[1:-1] + yO2[:-2]) / dz**2
    dyO2_dx = (yO2[1:-1] - yO2[:-2]) / dz
    
    dyN2_dt = np.zeros(Nz)
    dyO2_dt = np.zeros(Nz)
    
    # Mass Balance
    dyN2_dt[1:-1] = DL*d2yN2 - v*dyN2_dx - coupling*dqN2_dt[1:-1]
    dyO2_dt[1:-1] = DL*d2yO2 - v*dyO2_dx - coupling*dqO2_dt[1:-1]
    
    # BCs
    dyN2_dt[0] = (v/dz)*(1.0 - yN2[0]) - coupling*dqN2_dt[0]
    dyO2_dt[0] = (v/dz)*(0.0 - yO2[0]) - coupling*dqO2_dt[0]
    dyN2_dt[-1] = -v*(yN2[-1] - yN2[-2])/dz - coupling*dqN2_dt[-1]
    dyO2_dt[-1] = -v*(yO2[-1] - yO2[-2])/dz - coupling*dqO2_dt[-1]
    
    return np.concatenate([dyN2_dt, dyO2_dt, dqN2_dt, dqO2_dt])

def run_simulation(config, p_n2, p_o2, mode='real'):
    derived = calculate_physics(config, mode)
    Nz = config['Nz']
    
    # Initial Conditions: Pure O2
    y_init = np.zeros(Nz * 4)
    y_init[Nz:2*Nz] = 1.0
    y_init[3*Nz:4*Nz] = sips_model(config['P_tot'], *p_o2)
    
    # Use BDF solver. 
    # Important: If mode is ideal, system is VERY STIFF. BDF handles this.
    sol = solve_ivp(
        lambda t, y: pde_rhs(t, y, config, p_n2, p_o2, derived),
        (0, config['t_end']),
        y_init,
        method='BDF',
        rtol=1e-4, atol=1e-6
    )
    return sol

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("Fitting Isotherms...")
    p_Rom_N2 = fit_isotherm(Data_Rom_N2)
    p_Rom_O2 = fit_isotherm(Data_Rom_O2)
    p_Rem_N2 = fit_isotherm(Data_Rem_N2)
    p_Rem_O2 = fit_isotherm(Data_Rem_O2)
    
    materials = {
        "Romulus": (p_Rom_N2, p_Rom_O2, 'blue'),
        "Remus":   (p_Rem_N2, p_Rem_O2, 'red')
    }
    
    plt.figure(figsize=(10, 6))
    print("\nRunning Simulations (Real vs Ideal)...")
    
    for mat_name, (p_n2, p_o2, color) in materials.items():
        print(f" Processing {mat_name}...")
        
        # 1. Run Real (Standard Parameters)
        sol_real = run_simulation(CONFIG, p_n2, p_o2, mode='real')
        if sol_real.success:
            y_out = sol_real.y[CONFIG['Nz']-1, :]
            plt.plot(sol_real.t, y_out, color=color, linestyle='-', linewidth=2, label=f'{mat_name} (Real)')
            
        # 2. Run Ideal (DL=0, High k) via PDE Solver
        sol_ideal = run_simulation(CONFIG, p_n2, p_o2, mode='ideal')
        if sol_ideal.success:
            y_out_ideal = sol_ideal.y[CONFIG['Nz']-1, :]
            plt.plot(sol_ideal.t, y_out_ideal, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f'{mat_name} (Equilibrium)')
    
    plt.axhline(0.05, color='k', linestyle=':', alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Outlet N2 Mole Fraction")
    plt.title("Equilibrium Theory Comparison (Numerical Approximation)")
    plt.legend() 

#[Image of packed bed reactor column]

    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(CONFIG['outdir'], "Equilibrium_PDE_Only.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    plt.show()
