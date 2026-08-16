import numpy as np
import matplotlib.pyplot as plt

# Import your existing topology and risk-aware scheduler
from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.risk_profiles import inject_bs_failure
from src.scheduler3 import allocate_backup_paths  # Your explicit grouping scheduler
import src.constants as const

# Dummy evaluation function for structural testing
def mock_evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
    return True, np.random.uniform(0.5, 0.99)

def allocate_mc_baseline(affected_users, total_users, active_nodes, fixed_rb_value=10):
    """
    Simulates True Multi-Connectivity (MC).
    In MC, backup resources are pre-allocated 1:1 to ALL users in the network, 
    regardless of whether their primary link has failed or not.
    """
    if not affected_users:
        return 0, {}

    total_backup_rbs = len(active_nodes) * fixed_rb_value
    
    # Calculate the MC Protection Ratio. 
    mc_protection_ratio = min(1.0, total_backup_rbs / total_users)
    
    # Only the affected users who were lucky enough to secure a pre-allocated MC reservation are saved.
    recovered_count = int(len(affected_users) * mc_protection_ratio)

    return recovered_count, {}

def run_comparison_benchmark():
    """
    Tests MC vs Risk-Aware across scaling network loads.
    Simulates a Correlated Failure of 3 GBSs to trigger true disjoint grouping.
    """
    user_counts = [50, 100, 200, 300, 400, 500, 600, 700]
    
    mc_resilience = []
    ra_resilience = []
    
    fixed_rb_value = 10 
    num_gbs = 7
    
    # Correlated failure to allow for risk-disjoint pooling
    failed_bs_indices = [4, 5, 6] 
    
    print("Executing True MC vs. Risk-Aware Benchmark...")
    print(f"{'Total UEs':<10} | {'MC Resilience':<15} | {'Risk-Aware Resilience'}")
    print("-" * 55)
    
    for total_users in user_counts:
        mc_runs = []
        ra_runs = []
        
        # 20 Monte Carlo iterations per data point
        for seed in range(20):
            np.random.seed(seed)
            bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
            hap_coord, leo_coord = get_ntn_nodes()
            ue_coords = get_random_users(n=total_users)
            
            active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(num_gbs) if i not in failed_bs_indices]
            
            # Inject failure
            affected_users = inject_bs_failure(
                ue_coords, bs_coords, hap_coord, leo_coord, 
                failed_bs_indices, mock_evaluate_link
            )
            
            affected_count = len(affected_users)
            
            if affected_count == 0:
                mc_runs.append(100.0)
                ra_runs.append(100.0)
                continue
                
            # --- 1. Run True MC Baseline ---
            mc_recovered, _ = allocate_mc_baseline(affected_users, total_users, active_nodes, fixed_rb_value)
            mc_runs.append((mc_recovered / affected_count) * 100)
            
            # --- 2. Run Your Risk-Aware Scheduler ---
            # Unpacking 4 variables here so it doesn't crash
            ra_recovered, _, _, _ = allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value)
            ra_runs.append((ra_recovered / affected_count) * 100)
            
        mc_resilience.append(np.mean(mc_runs))
        ra_resilience.append(np.mean(ra_runs))
        
        print(f"{total_users:<10} | {mc_resilience[-1]:.1f}%{'':<10} | {ra_resilience[-1]:.1f}%")

    # Plot the Definitive Proof
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, ra_resilience, marker='o', linewidth=2.5, color='#2ca02c', label='Risk-Aware (Disjoint Grouping)')
    plt.plot(user_counts, mc_resilience, marker='s', linewidth=2.5, linestyle='--', color='#d62728', label='Multi-Connectivity (1:1 Hard Reservation)')
    
    plt.title('Network Resilience vs Total Load (Correlated 3-GBS Failure)', fontsize=14, pad=15)
    plt.xlabel('Total Users in Network', fontsize=12)
    plt.ylabel('Recovery Rate of Stranded Users (%)', fontsize=12)
    plt.xlim(40, 750)
    plt.ylim(0, 105)
    
    # Visual Annotations
    plt.axvline(x=60, color='grey', linestyle=':', alpha=0.6)
    plt.text(70, 50, 'MC Hardware Exhaustion\n(60 Total RBs)', color='black', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison_benchmark()