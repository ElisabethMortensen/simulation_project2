'''
Inpit Analysis: Estimate the distributions of each process and justify with one goodness-of-fit test
'''
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# load data
data = pd.read_excel("simulation_times_data.xlsx")
interarrival_times = data['Interarrival Times']
initial_phase = data['Service Times for Initial Phase']
placing_key_mouse = data['Service Times for Placing Keyboard and Mouse']
assemble_case = data['Service Times for Assembling the Case (Aluminum Plates)']

#Arrival Process
lambda_hat = 1 / interarrival_times.mean()
interarrival_params = (0, 1/lambda_hat)
print(f"Estimated rate (lambda) of the exponential distribution: {lambda_hat:.4f}")
ks_stat, p_value = stats.kstest(interarrival_times,'expon', args=(0, 1/lambda_hat))
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
# The p value is greater than 0.05, fail to reject null hypothesis 

#Manufacturing the Motherboard - Initial Phase
# Decided on lognorm based on histogram
initial_params = stats.lognorm.fit(initial_phase, floc=0)
print('Estimated parameters for lognorm distribution of initial phase:')
print(f"mu={np.log(initial_params[2]):.4f},sigma={initial_params[0]:.4f}")
ks_stat, p_val = stats.kstest(initial_phase, 'lognorm', args=initial_params)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_val:.4f}")

#Case Manufacturing - Placing the Keyboard and the Mouse
# Decided on lognorm based on histogram
km_params = stats.lognorm.fit(placing_key_mouse, floc=0)
print('Estimated parameters for lognorm distribution of placing keyboard and mouse:')
print(f"mu={np.log(km_params[2]):.4f}, sigma={km_params[0]:.4f}")
ks_stat, p_val = stats.kstest(placing_key_mouse, 'lognorm', args=km_params)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_val:.4f}")

## Main Assembly - Assembling the Case (Aluminum Plates)
# Decided on norm based on histogram
case_params  = stats.norm.fit(assemble_case)
print('Estimated parameters for normal distribution of assembling case:')
print(f"mean={case_params[0]:.4f},  std={case_params[1]:.4f}")
ks_stat, p_val = stats.kstest(assemble_case, 'norm', args=case_params)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_val:.4f}")

plot_config = [
    (interarrival_times, stats.expon, interarrival_params, "Interarrival Times", "Exponential"),
    (initial_phase, stats.lognorm,initial_params, "Initial Phase", "Log-Normal"),
    (placing_key_mouse,stats.lognorm,km_params, "Placing Keyboard & Mouse","Log-Normal"),
    (assemble_case,stats.norm,case_params, "Assembling the Case","Normal"),
]

# plottting each histogram with distribtuion line
for d, dist, params, title, dist_label in plot_config:
    x = np.linspace(max(d.min() * 0.5, 0), d.max() * 1.15, 300)
    _, ks_p = stats.kstest(d, dist.name, args=params)
    plt.figure(figsize=(7, 4))
    plt.hist(d, bins=25, density=True, alpha=0.4, color='steelblue', label='Data')
    plt.plot(x, dist.pdf(x, *params), 'r-', lw=2, label=f'{dist_label} fit')
    plt.title(f'{title}')
    plt.xlabel('Minutes')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()
