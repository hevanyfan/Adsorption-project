import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


plt.style.use('default')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})


#1. Column & Physics Parameters
epsilon = 0.32
rho_s = 990.0
rp = 1e-3
Tm = 303.0
Dm = 7.48e-5
tau = 3.0
R_gas = 8.314
Nz = 30  # Grid points

# 2. Experimental Data

# Romulus
P_rom_n2 = np.array([0, 2026.5, 3039.75, 4053.0, 6079.5, 8106.0, 61909.5, 111356.2, 173265.8, 247435.7, 321706.9, 395978.1, 470148.0, 544419.2, 618690.5])
q_rom_n2 = np.array([0, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835, 1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861])
P_rom_o2 = np.array([0, 8106.0, 20265.0, 247435.7, 371153.5, 494871.3, 618690.5])
q_rom_o2 = np.array([0, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815])

# Remus
P_rem_n2 = np.array([0, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000])
q_rem_n2 = np.array([0, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472])
P_rem_o2 = np.array([0, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700])
q_rem_o2 = np.array([0, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348])


# 3. Model & Fitting

def sips_model(P, qmax, b, n):
    P = np.maximum(P, 1e-6)
    val = (b * P) ** n
    return qmax * val / (1 + val)
# Fit
p0 = [2, 1e-4, 1]
bounds = ([0,0,0], [20,1,5])

# Get parameters
p_rom_n2, _ = curve_fit(sips_model, P_rom_n2, q_rom_n2, p0, bounds=bounds)
p_rom_o2, _ = curve_fit(sips_model, P_rom_o2, q_rom_o2, p0, bounds=bounds)
p_rem_n2, _ = curve_fit(sips_model, P_rem_n2, q_rem_n2, p0, bounds=bounds)
p_rem_o2, _ = curve_fit(sips_model, P_rem_o2, q_rem_o2, p0, bounds=bounds)


# 4. PDE System (Ghost Cells Preserved)

def ode_system_vectorized(t, y, L, u0, P_op, p_iso_n2, p_iso_o2, mode):
    
    # 1. Calculate Physics derived 
    Nz = len(y) // 4
    dx = L / (Nz - 1)
    
    v = u0 / epsilon
    
    # Mode selection
    if mode == 'equil':
        D = 0.0
        # Fast kinetics for equilibrium
        k = 15.0 * (Dm / tau) / (rp**2) * 100 
    else:
        D = 0.7 * Dm + v * rp  # Real dispersion
        k = 15.0 * (Dm / tau) / (rp**2) 
    #big term
    coupling = (rho_s * (1 - epsilon) / epsilon) * (R_gas * Tm / P_op)

    # 2. Unpack State
    yN2 = y[0:Nz]
    yO2 = y[Nz:2*Nz]
    qN2 = y[2*Nz:3*Nz]
    qO2 = y[3*Nz:4*Nz]

    # 3. Solid Phase (LDF)
    dqN2dt = k * (sips_model(yN2 * P_op, *p_iso_n2) - qN2)
    dqO2dt = k * (sips_model(yO2 * P_op, *p_iso_o2) - qO2)

    # 4. Gas Phase PDEs using Ghost Cells
    yN2_extended = np.concatenate(([1.0], yN2, [yN2[-1]])) 
    yO2_extended = np.concatenate(([0.0], yO2, [yO2[-1]])) 

    # Convection (Upwind)
    conv_N2 = v * (yN2_extended[:-2] - yN2_extended[1:-1]) / dx
    conv_O2 = v * (yO2_extended[:-2] - yO2_extended[1:-1]) / dx
    
    # Dispersion (Central)
    disp_N2 = D * (yN2_extended[:-2] - 2*yN2_extended[1:-1] + yN2_extended[2:]) / dx**2
    disp_O2 = D * (yO2_extended[:-2] - 2*yO2_extended[1:-1] + yO2_extended[2:]) / dx**2
    
    dyN2dt = disp_N2 + conv_N2 - coupling * dqN2dt
    dyO2dt = disp_O2 + conv_O2 - coupling * dqO2dt

    return np.concatenate((dyN2dt, dyO2dt, dqN2dt, dqO2dt))


# 5. Plotting


# Create figure 
fig, axes = plt.subplots(2, 2, figsize=(7, 5), dpi=300)
((ax1, ax2), (ax3, ax4)) = axes

# Base operating conditions
base_L = 0.32
base_u = 4.864e-3
base_P = 100e3

# PLOT A: Isotherms

P_range = np.linspace(0, 1000000, 100)

# Romulus
lbl_rom_n2 = f'Romulus N2 ($q_m$={p_rom_n2[0]:.2f}, b={p_rom_n2[1]:.1e}, n={p_rom_n2[2]:.2f})'
ax1.plot(P_range/1000, sips_model(P_range, *p_rom_n2), 'b-', label=lbl_rom_n2, linewidth=1)
ax1.scatter(P_rom_n2/1000, q_rom_n2, color='blue', marker='o', s=10) 

lbl_rom_o2 = f'Romulus O2 ($q_m$={p_rom_o2[0]:.2f}, b={p_rom_o2[1]:.1e}, n={p_rom_o2[2]:.2f})'
ax1.plot(P_range/1000, sips_model(P_range, *p_rom_o2), 'b--', label=lbl_rom_o2, linewidth=1)
ax1.scatter(P_rom_o2/1000, q_rom_o2, color='blue', marker='x', s=10)

# Remus
lbl_rem_n2 = f'Remus N2 ($q_m$={p_rem_n2[0]:.2f}, b={p_rem_n2[1]:.1e}, n={p_rem_n2[2]:.2f})'
ax1.plot(P_range/1000, sips_model(P_range, *p_rem_n2), 'r-', label=lbl_rem_n2, linewidth=1)
ax1.scatter(P_rem_n2/1000, q_rem_n2, color='red', marker='o', s=10)

lbl_rem_o2 = f'Remus O2 ($q_m$={p_rem_o2[0]:.2f}, b={p_rem_o2[1]:.1e}, n={p_rem_o2[2]:.2f})'
ax1.plot(P_range/1000, sips_model(P_range, *p_rem_o2), 'r--', label=lbl_rem_o2, linewidth=1)
ax1.scatter(P_rem_o2/1000, q_rem_o2, color='red', marker='x', s=10)

ax1.set_xlabel("Pressure (kPa)")
ax1.set_ylabel("Loading (mol/kg)")
ax1.set_title("A. Isotherms", fontsize=9)
ax1.legend(fontsize=5) 


# PLOT B: Equilibrium vs Real


# reset initial conditions
def get_init(P_val, iso_o2):
    y_init = np.zeros(4*Nz)
    y_init[Nz:2*Nz] = 1.0 
    y_init[3*Nz:4*Nz] = sips_model(P_val, *iso_o2) 
    return y_init

t_span = [0, 2000]
t_eval = np.linspace(0, 2000, 200)

# Romulus
y0 = get_init(base_P, p_rom_o2)
sol_real = solve_ivp(ode_system_vectorized, t_span, y0, t_eval=t_eval, method='BDF',
                     args=(base_L, base_u, base_P, p_rom_n2, p_rom_o2, 'real'))
ax2.plot(sol_real.t, sol_real.y[Nz-1], 'b-', lw=1.5, label='Romulus (Real)')

sol_eq = solve_ivp(ode_system_vectorized, t_span, y0, t_eval=t_eval, method='BDF',
                   args=(base_L, base_u, base_P, p_rom_n2, p_rom_o2, 'equil'))
ax2.plot(sol_eq.t, sol_eq.y[Nz-1], 'b--', lw=1, label='Romulus (Equil)')

# Remus
y0 = get_init(base_P, p_rem_o2)
sol_real = solve_ivp(ode_system_vectorized, t_span, y0, t_eval=t_eval, method='BDF',
                     args=(base_L, base_u, base_P, p_rem_n2, p_rem_o2, 'real'))
ax2.plot(sol_real.t, sol_real.y[Nz-1], 'r-', lw=1.5, label='Remus (Real)')

sol_eq = solve_ivp(ode_system_vectorized, t_span, y0, t_eval=t_eval, method='BDF',
                   args=(base_L, base_u, base_P, p_rem_n2, p_rem_o2, 'equil'))
ax2.plot(sol_eq.t, sol_eq.y[Nz-1], 'r--', lw=1, label='Remus (Equil)')

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Outlet N2")
ax2.set_title("B. Equilibrium Comparison", fontsize=9)
ax2.legend(fontsize=6) 


# PLOT C: Sensitivity - ROMULUS

t_sens = np.linspace(0, 3500, 300)
t_span_sens = [0, 3500]
iso_n = p_rom_n2
iso_o = p_rom_o2
ax = ax3
ax.set_title("C. Sensitivity - Romulus", fontsize=9)

# 1. Base Case
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '-', label='Base Case', linewidth=1)

# 2. Low Velocity
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u*0.5, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '--', label='Low Velocity (-50%)', linewidth=1)

# 3. High Velocity
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u*1.5, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '--', label='High Velocity (+50%)', linewidth=1)

# 4. Short Column
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(0.2, base_u, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '-.', label='Short Column (0.2m)', linewidth=1)

# 5. Long Column
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(0.5, base_u, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '-.', label='Long Column (0.5m)', linewidth=1)

# 6. High Pressure
y0 = get_init(2e5, iso_o) 
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u, 2e5, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], ':', label='High Pressure (2 bar)', linewidth=1)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Outlet N2")
ax.set_xlim(0, 3500)
# [MODIFIED: 2 cols to fit in small width, tiny font]
ax.legend(fontsize=5, ncol=2) 


# PLOT D: Sensitivity - REMUS

iso_n = p_rem_n2
iso_o = p_rem_o2
ax = ax4
ax.set_title("D. Sensitivity - Remus", fontsize=9)

# 1. Base Case
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '-', label='Base Case', linewidth=1)

# 2. Low Velocity
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u*0.5, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '--', label='Low Velocity (-50%)', linewidth=1)

# 3. High Velocity
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u*1.5, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '--', label='High Velocity (+50%)', linewidth=1)

# 4. Short Column
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(0.2, base_u, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '-.', label='Short Column (0.2m)', linewidth=1)

# 5. Long Column
y0 = get_init(base_P, iso_o)
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(0.5, base_u, base_P, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], '-.', label='Long Column (0.5m)', linewidth=1)

# 6. High Pressure
y0 = get_init(2e5, iso_o) 
sol = solve_ivp(ode_system_vectorized, t_span_sens, y0, t_eval=t_sens, method='BDF',
                args=(base_L, base_u, 2e5, iso_n, iso_o, 'real'))
ax.plot(sol.t, sol.y[Nz-1], ':', label='High Pressure (2 bar)', linewidth=1)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Outlet N2")
ax.set_xlim(0, 3500)
ax.legend(fontsize=5, ncol=2)


plt.tight_layout(pad=0.5) 
plt.savefig('ASP_Final_Figure.png', dpi=300, bbox_inches='tight')
plt.show()
