import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calculate_total_noise_unitary(n: int, lambda_idle: float, lambda_cnot: float, lambda_meas: float) -> float:
    t_idle = n**2/4-3*n/2+2
    N_cnot = n-1
    N_meas = 0

    lambda_tot = t_idle*lambda_idle + N_cnot*lambda_cnot + N_meas*lambda_meas
    return lambda_tot

def calculate_total_noise_dynamic(n: int, lambda_idle: float, lambda_cnot: float, lambda_meas: float, mu: float) -> float:
    t_idle = 1+n/2*mu
    N_cnot = 3*n/2-2
    N_meas = n/2-1  

    lambda_tot = t_idle*lambda_idle + N_cnot*lambda_cnot + N_meas*lambda_meas
    return lambda_tot

def fidelity_lower_bound_unitary(n_list: np.ndarray, lambda_idle: float, lambda_cnot: float, lambda_meas: float) -> np.ndarray:
    lam_tot_list = np.asarray([calculate_total_noise_unitary(n, lambda_idle, lambda_cnot, lambda_meas) for n in n_list])
    return np.exp(-lam_tot_list)

def fidelity_lower_bound_dynamic(n_list: np.ndarray, lambda_idle: float, lambda_cnot: float, lambda_meas: float) -> np.ndarray:
    # ibm prekskill hardware data
    t_meas = 0.9 # mus
    t_ff = 0.7 # mus
    t_cnot = 0.6 # mus

    lam_tot_list = []
    for n in n_list:
        N_meas = n/2-1
        mu = (t_meas+t_ff*2**(N_meas-1))/t_cnot
    
        lam_tot_list.append(calculate_total_noise_dynamic(n, lambda_idle, lambda_cnot, lambda_meas, mu))
    lam_tot_list = np.asarray(lam_tot_list)
    return np.exp(-lam_tot_list)

def fidelity_bound_dynamic_fac(lambda_idle: float, lambda_cnot: float):
    def fidelity_lower_bound_dynamic(n_list: np.ndarray, lambda_meas: float) -> np.ndarray:
        # ibm prekskill hardware data
        t_meas = 0.9 # mus
        t_ff = 0.7 # mus
        t_cnot = 0.6 # mus

        lam_tot_list = []
        for n in n_list:
            N_meas = n/2-1
            mu = (t_meas+t_ff*2**(N_meas-1))/t_cnot
        
            lam_tot_list.append(calculate_total_noise_dynamic(n, lambda_idle, lambda_cnot, lambda_meas, mu))
        lam_tot_list = np.asarray(lam_tot_list)
        return np.exp(-lam_tot_list)
    
    return fidelity_lower_bound_dynamic



def fidelity_lower_bound_combined(n_list_comb: np.ndarray, lambda_idle: float, lambda_cnot: float, lambda_meas: float) -> np.ndarray:
    n_list_uni = n_list_comb[:10] # len(xdata_uni) = 10
    n_list_dyn = n_list_comb[10:]
    return np.concatenate([fidelity_lower_bound_unitary(n_list_uni, lambda_idle, lambda_cnot, lambda_meas), fidelity_lower_bound_dynamic(n_list_dyn, lambda_idle, lambda_cnot, lambda_meas)])

if __name__=="__main__":
    # fit lambda rates to data extracted from ibm plot
    xdata_uni = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    ydata_uni = [0.98, 0.94, 0.90, 0.85, 0.8, 0.73, 0.68, 0.62, 0.55, 0.5]

    xdata_dyn = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    ydata_dyn = [0.9, 0.8, 0.69, 0.6, 0.51, 0.418, 0.32, 0.21, 0.1, 0.02]

    # # Fit the function to the data
    # popt_uni, pcov_uni = curve_fit(fidelity_lower_bound_unitary, xdata_uni, ydata_uni)
    # lam_idle_opt, lam_cnot_opt, lam_meas_opt = popt_uni
    # print(f"Optimal parameters for unitary fit: lambda_idle={lam_idle_opt:.4f}, lambda_cnot={lam_cnot_opt:.4f}, lambda_meas={lam_meas_opt:.4f}")

    # popt_dyn, pcov_dyn = curve_fit(fidelity_lower_bound_dynamic, xdata_dyn, ydata_dyn)
    # lam_idle_opt, lam_cnot_opt, lam_meas_opt = popt_dyn
    # print(f"Optimal parameters for dynamic fit: lambda_idle={lam_idle_opt:.4f}, lambda_cnot={lam_cnot_opt:.4f}, lambda_meas={lam_meas_opt:.4f}")

    # Combine the x-data into a single array
    x_combined = np.concatenate([xdata_uni, xdata_dyn])
    y_combined = np.concatenate([ydata_uni, ydata_dyn])
    
    # Fit the shared parameters
    popt, pcov = curve_fit(fidelity_lower_bound_combined, x_combined, y_combined)
    
    # Extract optimized parameters
    lam_idle_opt, lam_cnot_opt, lam_meas_opt = popt
    print(f"Optimal parameters for combined fit: lambda_idle={lam_idle_opt:.4f}, lambda_cnot={lam_cnot_opt:.4f}, lambda_meas={lam_meas_opt:.4f}")

    # fit_fctn_dyn = fidelity_bound_dynamic_fac(lam_idle_opt, lam_cnot_opt)
    # popt_dyn, pcov_dyn = curve_fit(fit_fctn_dyn, xdata_dyn, ydata_dyn)
    # lam_meas_opt = popt_dyn[0]
    # print(f"Optimal parameters for unitary fit: lambda_idle={lam_idle_opt:.4f}, lambda_cnot={lam_cnot_opt:.4f}, lambda_meas={lam_meas_opt:.4f}")


    # Add ibm prekskill hardware data
    #lambda_cnot = 5*10**(-3)
    #lambda_meas = 1.2*10**(-2)
    # lambda_cnot = 0.02
    # lambda_meas = 0.012
    # lambda_idle = 0.001
    t_meas = 0.9 # mus
    t_ff = 0.7 # mus
    t_cnot = 0.6 # mus
    ## To-Do: this should exponentially depend to the number of measurements?
    #mu = (t_meas+t_ff)/t_cnot

    lambda_cnot = lam_cnot_opt
    lambda_meas = lam_meas_opt
    lambda_idle = lam_idle_opt

    n_list = np.linspace(3, 21, 19)

    fidelities_dynamic = []
    fidelities_unitary = []
    for n in n_list:
        lam_tot_unitary = calculate_total_noise_unitary(n, lambda_idle, lambda_cnot, lambda_meas)
        fidelities_unitary.append(math.exp(-lam_tot_unitary))

        N_meas = n/2-1
        mu = (t_meas+t_ff*2**(N_meas-1))/t_cnot
        lam_tot_dynamic = calculate_total_noise_dynamic(n, lambda_idle, lambda_cnot, lambda_meas, mu)
        fidelities_dynamic.append(math.exp(-lam_tot_dynamic))

    plt.figure()
    plt.plot(n_list, fidelities_unitary, label="unitary", color='tab:blue')
    plt.scatter(xdata_uni, ydata_uni, label="unitary, data", color='tab:blue', zorder=5)
    plt.plot(n_list, fidelities_dynamic, label="dynamic", color='tab:orange')
    plt.scatter(xdata_dyn, ydata_dyn, label="dynamic, data", color='tab:orange', zorder=5)
    plt.xlabel("Number of qubits")
    plt.ylabel("Fidelity")
    plt.ylim(0.0, 1.0)
    plt.xlim(3, 21)
    plt.grid(which='both', axis='both')
    plt.xticks(np.arange(3, 22, 2))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.legend()
    plt.savefig("error_budget_ibm_fidelity.pdf", bbox_inches="tight")
    plt.show()

