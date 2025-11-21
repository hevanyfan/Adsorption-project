# Advanced Separation Processes - Part 2: Breakthrough Simulation
# Course: CHEN40461/60461 - Advanced Separation Processes AY 2025-26
# Run in Jupyter: %run asp_breakthrough_model.py
# Or terminal:    python asp_breakthrough_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. CONFIGURATION & PARAMETERS (Table 2)
# ==============================================================================

CONFIG = {
    # Column Dimensions
    'L': 0.32,          # m (Column Length)
    'epsilon': 0.32,    # Void Fraction (omega)
    'rho_s': 990.0,     # kg/m3 (Particle Density)
    'r_p': 1e-3,        # m (Particle Radius)
    
    # Operating Conditions
    'T': 303.0,         # K
    'P_tot': 100e3,     # Pa (100 kPa)
    'u_in': 4.864e-3,   # m/s (Superficial Velocity)
    
    # Transport Properties
    'D_m': 7.48e-5,     # m2/s (Molecular Diffusivity)
    'tau': 3.0,         # Tortuosity
    'R_gas': 8.314,     # Gas Constant
    
    # Numerical Settings
    'Nz': 30,           # Grid points (30 is robust/fast)
    't_end': 2500.0,    # End time (s)
    'outdir': 'outputs' # Output folder
}

# Create output folder if it doesn't exist
os.makedirs(CONFIG['outdir'], exist_ok=True)

# ==============================================================================
# 2. EXPERIMENTAL DATA (Table 1)
# ==============================================================================

# Romulus Isotherm Data
Data_Rom_N2 = {
    'P': np.array([0, 2026.5, 3039.75, 4053.0, 6079.5, 8106.0, 61909.5, 111356.2, 173265.8, 247435.7, 321706.9, 395978.1, 470148.0, 544419.2, 618690.5]),
    'q': np.array([0, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835, 1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861])
}
Data_Rom_O2 = {
    'P': np.array([0, 8106.0, 20265.0, 247435.7, 371153.5, 494871.3, 618690.5]),
    'q': np.array([0, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815])
}

# Remus Isotherm Data
Data_Rem_N2 = {
    'P': np.array([0, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000]),
    'q': np.array([0, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472])
}
Data_Rem_O2 = {
    'P': np.array([0, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700]),
    'q': np.array([0, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348])
}

# ==============================================================================
# 3. ISOTHERM MODELING (Sips Equation)
# ==============================================================================

def sips_model(P, qmax, b, n):
    """
    Sips Isotherm: q = qmax * (bP)^n / (1 + (bP)^n)
    Includes safety clips to prevent numerical errors with negative pressures.
    """
    P_safe = np.maximum(P, 0.0)
    bP = b * P_safe
    
    # Handle potential overflow for large P
    try:
        term = bP**n
    except:
        term = np.full_like(P, 1e10) 
        
    return qmax * term / (1.0 + term)

def fit_isotherm(data_dict):
    """Fits the Sips model to P vs q data."""
    # Initial guess: qmax=2, b=1e-5, n=0.9
    p0 = [2.0, 1e-5, 0.9]
    # Bounds to keep physics realistic
    bounds = ([0, 0, 0.1], [20, 1.0, 5.0])
    
    params, _ = curve_fit(sips_model, data_dict['P'], data_dict['q'], 
                          p0=p0, bounds=bounds, maxfev=10000)
    return params

# ==============================================================================
# 4. BREAKTHROUGH SIMULATION (PDE Engine)
# ==============================================================================

def calculate_derived_params(cfg):
    """Calculates v, DL, and k_LDF based on current config."""
    # Interstitial Velocity (v = u / epsilon)
    v_inter = cfg['u_in'] / cfg['epsilon']
    
    # Axial Dispersion
    DL = 0.7 * cfg['D_m'] + v_inter * cfg['r_p']
    
    # Mass Transfer Coefficient (Glueckauf LDF)
    Dp = cfg['D_m'] / cfg['tau']
    k_LDF = 15.0 * Dp / (cfg['r_p']**2)
    
    # Coupling Factor for PDE: (RT/P) * (rho * (1-e)/e)
    term = (cfg['rho_s'] * (1.0 - cfg['epsilon']) / cfg['epsilon'])
    coupling = (cfg['R_gas'] * cfg['T'] / cfg['P_tot']) * term
    
    return v_inter, DL, k_LDF, coupling

def pde_rhs(t, y_flat, cfg, params_n2, params_o2, derived_params):
    """Discretized PDE equations."""
    v, DL, k, coupling = derived_params
    Nz = cfg['Nz']
    dz = cfg['L'] / (Nz - 1)
    
    # Unpack state vector
    yN2 = y_flat[0:Nz]
    yO2 = y_flat[Nz:2*Nz]
    qN2 = y_flat[2*Nz:3*Nz]
    qO2 = y_flat[3*Nz:4*Nz]
    
    # SAFETY: Clip mole fractions 0-1
    yN2 = np.clip(yN2, 0.0, 1.0)
    yO2 = np.clip(yO2, 0.0, 1.0)
    
    # 1. Solid Phase (LDF)
    qN2_star = sips_model(yN2 * cfg['P_tot'], *params_n2)
    qO2_star = sips_model(yO2 * cfg['P_tot'], *params_o2)
    
    dqN2_dt = k * (qN2_star - qN2)
    dqO2_dt = k * (qO2_star - qO2)
    
    # 2. Gas Phase (Convection + Dispersion + Adsorption)
    dyN2_dt = np.zeros(Nz)
    dyO2_dt = np.zeros(Nz)
    
    # Gradients
    d2yN2 = (yN2[2:] - 2*yN2[1:-1] + yN2[:-2]) / dz**2
    dyN2_dx = (yN2[1:-1] - yN2[:-2]) / dz # Upwind
    
    d2yO2 = (yO2[2:] - 2*yO2[1:-1] + yO2[:-2]) / dz**2
    dyO2_dx = (yO2[1:-1] - yO2[:-2]) / dz # Upwind
    
    # Internal Nodes
    dyN2_dt[1:-1] = DL*d2yN2 - v*dyN2_dx - coupling*dqN2_dt[1:-1]
    dyO2_dt[1:-1] = DL*d2yO2 - v*dyO2_dx - coupling*dqO2_dt[1:-1]
    
    # Boundary: Inlet (z=0, Danckwerts)
    yN2_in, yO2_in = 1.0, 0.0
    dyN2_dt[0] = (v/dz)*(yN2_in - yN2[0]) - coupling*dqN2_dt[0]
    dyO2_dt[0] = (v/dz)*(yO2_in - yO2[0]) - coupling*dqO2_dt[0]
    
    # Boundary: Outlet (z=L, Neumann)
    dyN2_dt[-1] = -v*(yN2[-1] - yN2[-2])/dz - coupling*dqN2_dt[-1]
    dyO2_dt[-1] = -v*(yO2[-1] - yO2[-2])/dz - coupling*dqO2_dt[-1]
    
    return np.concatenate([dyN2_dt, dyO2_dt, dqN2_dt, dqO2_dt])

def run_simulation(config, p_n2, p_o2):
    """Wrapper to prepare ICs and run solve_ivp."""
    derived = calculate_derived_params(config)
    Nz = config['Nz']
    
    # Initial Conditions: Bed saturated with O2
    y_init = np.zeros(Nz * 4)
    y_init[Nz:2*Nz] = 1.0 # yO2 = 1.0
    
    # Initial qO2 in equilibrium
    qO2_sat = sips_model(config['P_tot'], *p_o2)
    y_init[3*Nz:4*Nz] = qO2_sat
    
    # Solve (using BDF for stiffness)
    sol = solve_ivp(
        lambda t, y: pde_rhs(t, y, config, p_n2, p_o2, derived),
        (0, config['t_end']),
        y_init,
        method='BDF',
        rtol=1e-4, atol=1e-6
    )
    return sol

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    print("--- 1. Fitting Isotherms ---")
    # Fit Parameters
    p_Rom_N2 = fit_isotherm(Data_Rom_N2)
    p_Rom_O2 = fit_isotherm(Data_Rom_O2)
    p_Rem_N2 = fit_isotherm(Data_Rem_N2)
    p_Rem_O2 = fit_isotherm(Data_Rem_O2)
    
    print(f"Romulus N2: {p_Rom_N2}")
    print(f"Remus N2:   {p_Rem_N2}")

    print("\n--- 2. Running Sensitivity Analysis (Both Materials) ---")
    
    # Define Scenarios: (Name, velocity, Length, Pressure)
    # Base: u=0.004864, L=0.32, P=100e3
    scenarios = [
        ("Base Case", 4.864e-3, 0.32, 100e3),
        ("Low Velocity (-50%)", 2.432e-3, 0.32, 100e3),
        ("High Velocity (+50%)", 7.296e-3, 0.32, 100e3),
        ("Short Column (0.2m)", 4.864e-3, 0.2, 100e3),
        ("Long Column (0.5m)", 4.864e-3, 0.5, 100e3),
        ("High Pressure (2 bar)", 4.864e-3, 0.32, 200e3)
    ]
    
    # Materials Dictionary for Looping
    materials = {
        "Romulus": (p_Rom_N2, p_Rom_O2),
        "Remus":   (p_Rem_N2, p_Rem_O2)
    }
    
    # Create plots for each material
    for mat_name, (p_n2, p_o2) in materials.items():
        print(f"Simulating Material: {mat_name}...")
        
        plt.figure(figsize=(10, 6))
        
        for scen_name, u_val, L_val, P_val in scenarios:
            # Create temporary config for this scenario
            curr_cfg = CONFIG.copy()
            curr_cfg['u_in'] = u_val
            curr_cfg['L'] = L_val
            curr_cfg['P_tot'] = P_val
            
            # Increase time for slow cases (Low velocity or Long column)
            if u_val < 4e-3 or L_val > 0.4:
                curr_cfg['t_end'] = 3000
            else:
                curr_cfg['t_end'] = 2500
            
            # Run
            sol = run_simulation(curr_cfg, p_n2, p_o2)
            
            # Extract Outlet N2 (Node Nz-1)
            if sol.success:
                y_out = sol.y[curr_cfg['Nz']-1, :]
                
                # Styling
                style = '-'
                if "Velocity" in scen_name: style = '--'
                elif "Column" in scen_name: style = '-.'
                elif "Pressure" in scen_name: style = ':'
                
                plt.plot(sol.t, y_out, linestyle=style, linewidth=2, label=scen_name)
            else:
                print(f"  [Failed] {scen_name}")

        # Finalize Plot
        plt.axhline(0.05, color='k', alpha=0.2, label='Breakthrough (5%)')
        plt.xlabel("Time (s)")
        plt.ylabel("Outlet N2 Mole Fraction")
        plt.title(f"Sensitivity Analysis - {mat_name}")
        plt.legend(loc='lower right', fontsize='small', ncol=2)
        plt.grid(True, alpha=0.3)
        
        # Save
        filename = os.path.join(CONFIG['outdir'], f"Sensitivity_{mat_name}.png")
        plt.savefig(filename, dpi=150)
        print(f"  Saved plot to {filename}")
        plt.show()

    print("\nDone.")
