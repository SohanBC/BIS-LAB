RESEARCH PAPER:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7379495&tag=1

PROBLEM STATEMENT:  
Traffic signal optimisation
Static traffic signals waste commuter time and fuel by ignoring real-time road conditions, creating the need for an adaptive system that dynamically optimizes green-light durations to minimize intersection congestion.

IMPLEMENTATION CODE :  

```py

import numpy as np
import pandas as pd
import random
from scipy.special import gamma

# =====================================================================
# 1. SETUP & REPRODUCIBILITY
# =====================================================================
np.random.seed(42)
random.seed(42)

# =====================================================================
# 2. PROBLEM PARAMETERS & OBJECTIVE FUNCTION
# =====================================================================
NUM_PHASES = 4
MIN_GREEN = 10   # Minimum green time for safety
MAX_GREEN = 60   # Maximum green time to prevent starvation
CYCLE_TIME = 120 # Fixed total cycle time

# Arrival rates (vehicles per second) for each phase
ARRIVAL_RATES = np.array([0.2, 0.15, 0.25, 0.1]) 
# Saturation flow rates (vehicles per second)
SATURATION_FLOW = np.array([1.0, 1.0, 1.0, 1.0])

def objective_function(green_times):
    """
    Calculates the total average vehicle delay using a simplified Webster-like delay model.
    """
    clearance_time = NUM_PHASES * 4 # 4 seconds lost time per phase
    available_green = CYCLE_TIME - clearance_time
    
    sum_green = np.sum(green_times)
    if sum_green == 0:
        return float('inf')
    
    scaled_green = (green_times / sum_green) * available_green
    scaled_green = np.clip(scaled_green, MIN_GREEN, MAX_GREEN)
    
    sum_green = np.sum(scaled_green)
    scaled_green = (scaled_green / sum_green) * available_green
    
    total_delay = 0
    for i in range(NUM_PHASES):
        g = scaled_green[i]
        c = CYCLE_TIME
        lambda_i = ARRIVAL_RATES[i]
        s = SATURATION_FLOW[i]
        
        capacity = s * (g / c)
        degree_of_sat = lambda_i / capacity if capacity > 0 else 1.0
        
        if degree_of_sat >= 1.0:
            delay = 0.5 * c * (1 - g/c)**2 / (1 - min(0.99, lambda_i/s)) + 1000 * (degree_of_sat - 1.0)
        else:
            delay = 0.5 * c * (1 - g/c)**2 / (1 - lambda_i/s)
            
        total_delay += delay * lambda_i
        
    return total_delay

# =====================================================================
# 3. CUCKOO SEARCH ALGORITHM IMPLEMENTATION
# =====================================================================
def levy_flight(Lambda, size):
    """Generates steps from a Levy flight distribution."""
    temp = gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / (gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2))
    sigma_u = temp**(1 / Lambda)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, size=size)
    v = np.random.normal(0, sigma_v, size=size)
    step = u / (np.abs(v)**(1 / Lambda))
    return step

def cuckoo_search_traffic(n_nests=15, max_iter=10, pa=0.25):
    nests = np.random.uniform(MIN_GREEN, MAX_GREEN, (n_nests, NUM_PHASES))
    fitness = np.array([objective_function(nest) for nest in nests])
    
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    history = []
    
    print(f"Initial Best Delay: {best_fitness:.2f} seconds | Green Phases: {np.round(best_nest, 1)}")
    print("-" * 80)
    
    for iteration in range(1, max_iter + 1):
        new_nests = nests.copy()
        step_size = levy_flight(1.5, (n_nests, NUM_PHASES))
        
        for i in range(n_nests):
            step = step_size[i] * 0.01 * (nests[i] - best_nest)
            new_nests[i] += np.random.normal(0, 1, NUM_PHASES) * step
            new_nests[i] = np.clip(new_nests[i], MIN_GREEN, MAX_GREEN)
            
            f_new = objective_function(new_nests[i])
            random_partner_idx = np.random.randint(0, n_nests)
            if f_new < fitness[random_partner_idx]:
                nests[random_partner_idx] = new_nests[i]
                fitness[random_partner_idx] = f_new
        
        for i in range(n_nests):
            if np.random.rand() > pa:
                step_size_abandon = np.random.rand() * (nests[np.random.randint(0, n_nests)] - nests[np.random.randint(0, n_nests)])
                nests[i] += step_size_abandon
                nests[i] = np.clip(nests[i], MIN_GREEN, MAX_GREEN)
                fitness[i] = objective_function(nests[i])
                
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()
            
        history.append((iteration, best_fitness, best_nest.copy()))
        print(f"Iteration {iteration:02d} | Best Delay: {best_fitness:.4f}s | Optimal Green Times: {np.round(best_nest, 1)}")
        
    return best_nest, best_fitness, history

# =====================================================================
# 4. RUN OPTIMIZATION
# =====================================================================
best_solution, best_score, logs = cuckoo_search_traffic(n_nests=20, max_iter=10, pa=0.25)

# =====================================================================
# 5. VISUALIZATION
# =====================================================================
import matplotlib.pyplot as plt

# Extract data for plotting from the logs
iterations = [log[0] for log in logs]
best_delays = [log[1] for log in logs]

# The objective function scales the raw Cuckoo Search output to fit the available cycle time. 
# We need to apply that same scaling to the best_solution to see the *actual* final green times.
clearance_time = NUM_PHASES * 4 
available_green = CYCLE_TIME - clearance_time
scaled_green = (best_solution / np.sum(best_solution)) * available_green
scaled_green = np.clip(scaled_green, MIN_GREEN, MAX_GREEN)
scaled_green = (scaled_green / np.sum(scaled_green)) * available_green

phase_labels = [f"Phase {i+1}\n(λ={ARRIVAL_RATES[i]})" for i in range(NUM_PHASES)]

# Create a figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Graph 1: Convergence Curve ---
ax1.plot(iterations, best_delays, marker='o', linestyle='-', color='b', linewidth=2)
ax1.set_title('Cuckoo Search Optimization Convergence', fontsize=12, fontweight='bold')
ax1.set_xlabel('Iteration', fontsize=10)
ax1.set_ylabel('Total Average Delay (seconds)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Graph 2: Final Green Time Allocations ---
bars = ax2.bar(phase_labels, scaled_green, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
ax2.set_title('Optimized Green Time Allocation per Phase', fontsize=12, fontweight='bold')
ax2.set_xlabel('Traffic Phases (λ = Arrival Rate)', fontsize=10)
ax2.set_ylabel('Green Time (seconds)', fontsize=10)
ax2.set_ylim(0, max(scaled_green) + 10) # Add a little headroom for labels
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add exact second values on top of the bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}s',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


```

<img width="715" height="220" alt="image" src="https://github.com/user-attachments/assets/0443a168-1886-4eba-a61a-e64494976287" />
<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/c417ff74-06d1-4412-bc45-1e929c6829d0" />
