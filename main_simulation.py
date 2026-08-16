import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, 
    sinr, rate, error_probability, check_transmission_success
)
import src.constants as const
from src.risk_profiles import inject_bs_failure
from src.scheduler_final import allocate_backup_paths

from src.link_evaluator import get_all_link_budgets
from src.primary_path import evaluate_primary_connection
from src.mc_scheme import apply_mc_scheme
from src.visualization import plot_topology

# # Set a default global reference for the core functions to use
# from src.mc_evaluator import evaluate_multi_connectivity


def evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
    # 1. Calculate Capacity
    # (Using system bandwidth here assumes p_tx is the total node transmit power)
    sinr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_RB)
    
    MAX_SE = 8.0 # Maximum Spectral Efficiency of 8 bps/Hz with 256-QAM modulation
    capped_se  = min(MAX_SE, math.log2(1 + sinr_lin))
    
    # 2. Calculate Reliability (psi_s) for each segment
    ep_wireless = error_probability(sinr_lin, const.MODULATION_M)
    psi_wireless = 1 - ep_wireless
    psi_backhaul = 1 - const.BACKHAUL_ERROR_PROB
     
    # 3. Calculate E2E Availability 
    """For simplicty physical availablity (rho) is chosen to be unity (that means, up-time is 100% in normal conditon) and 
    overall a_jn depends on reliability of segment (psi_s)"""
    a_jn = (psi_wireless * rho_wireless) * (psi_backhaul * rho_backhaul)
    
    #sinr_db = 10 * math.log10(sinr_lin) if sinr_lin > 0 else 0.0
        
    return a_jn, capped_se, dist


def run_simulation(num_users=const.NUM_UE, num_gbs=const.NUM_GBS, fixed_rb_value=10, seed_val=None, verbose=True):
    if seed_val is not None:
        np.random.seed(seed_val)
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=num_users)
    
    failed_bs_indices = [4,5,6] # Three ground base stations failed
    active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(num_gbs) if i not in failed_bs_indices]
    
    affected_users = inject_bs_failure(
        ue_coords, bs_coords, hap_coord, leo_coord, 
        failed_bs_indices, evaluate_link
    )
    
    # Output from scheduler
    recovered_count, allocated_loads, node_composition, assigned_scores, recovery_details = allocate_backup_paths(
        affected_users=affected_users, 
        active_nodes=active_nodes, 
        failed_bs_indices=failed_bs_indices,
        fixed_rb_value=fixed_rb_value,
        verbose=verbose
    )

    affected_count = len(affected_users)
    resilience = ((recovered_count / affected_count) * 100) if affected_count > 0 else 100.0
    
    return affected_count, resilience, allocated_loads, node_composition, assigned_scores, recovery_details


def plot_risk_disjoint_proof(node_composition):
    """A function to show the distrubution of recovered users among the backup nodes"""
    backup_nodes = ['HAP', 'LEO', 'GBS_0', 'GBS_1', 'GBS_2', 'GBS_3']
    
    all_primaries = set()
    for node in backup_nodes:
        if node in node_composition:
            all_primaries.update(node_composition[node].keys())
    
    all_primaries = sorted(list(all_primaries))
    
    node_names = backup_nodes
    bottoms = np.zeros(len(node_names))
    
    plt.figure(figsize=(8, 6))
    
    for primary in all_primaries:
        values = [node_composition.get(node, {}).get(primary, 0) for node in node_names]
        plt.bar(node_names, values, bottom=bottoms, label=f'Failed {primary}')
        bottoms += values

    plt.title('Risk-Disjoint Backup Allocation Across Backup Nodes', fontsize=14, pad=15)
    plt.ylabel('Number of Allocated Users', fontsize=12)
    plt.yticks(range(0, int(max(bottoms)) + 5, 2))
    plt.legend(title="Original Primary Node")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def run_monte_carlo_averaging(user_counts, num_gbs=7, fixed_rbs=10, runs_per_scenario=50):
    user_counts = [50,100,200,300,400]
    resilience_results = []
    
    global_affected = 0
    global_recovered = 0
    
    print(f"\nMonte Carlo Simulation ({runs_per_scenario} Runs/Point) | {num_gbs} GBS | {fixed_rbs} RBs")
    print(f"{'Total UEs':<10} | {'Affected':<10} | {'Recovered':<10} | {'HAP':<6} | {'LEO':<6} | {'GBS':<6} | {'Network Resilience'}")
    
    for total_users in user_counts:
        runs = [run_simulation(num_users=total_users, num_gbs=num_gbs, fixed_rb_value=fixed_rbs, seed_val=i, verbose=False)
                for i in range(runs_per_scenario)]
        
         
        avg_affected = round(np.mean([r[0] for r in runs]))
        avg_resilience = np.mean([r[1] for r in runs])
        avg_recovered = round(np.mean([r[0] * (r[1] / 100) for r in runs]))
        
        avg_hap = round(np.mean([r[2].get('HAP', 0) for r in runs]))
        avg_leo = round(np.mean([r[2].get('LEO', 0) for r in runs]))
        
        avg_gbs = avg_recovered - (avg_hap + avg_leo)
        
        resilience_results.append(avg_resilience)
        
        global_affected += sum(r[0] for r in runs)
        global_recovered += sum(r[0] * (r[1] / 100) for r in runs)
        
        print(f"{total_users:<10} | {avg_affected:<10} | {avg_recovered:<10} | {avg_hap:<6} | {avg_leo:<6} | {avg_gbs:<6} | {avg_resilience:.2f}%")

    weighted_avg = ((global_recovered / global_affected) * 100) if global_affected > 0 else 100.0
    print(f"TRUE WEIGHTED AVERAGE RESILIENCE : {weighted_avg:.2f}%\n")
    
    return resilience_results


def generate_report(total_users, res_10_rb, res_20_rb):
    plt.figure(figsize=(10, 6))
    plt.plot(total_users, res_10_rb, marker='o', linestyle='-', color='#1f77b4', label='10 RBs per Backup Node')
    plt.plot(total_users, res_20_rb, marker='s', linestyle='-', color='#ff7f0e', label='20 RBs per Backup Node')
    plt.title('Average Network Resilience vs. Total Users (7 GBS Topology)', fontsize=14, pad=15)
    plt.xlabel('Total Users in Network', fontsize=12)
    plt.ylabel('Network Resilience (%)', fontsize=12)
    plt.xlim(40, 410)
    plt.ylim(0, 105) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()
    
    
def evaluate_mc_scheme():
    """Function to run primary and MC evaluation"""
    print("\n" + "="*60)
    print("RUNNING PRIMARY BASELINE & MULTI-CONNECTIVITY ")
    print("="*60)
    
    # 1. Generate Topology
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=const.NUM_UE)
    
    # 2. Run Primary Baseline
    print("Evaluating Physical Links and Primary Baseline...")
    primary_df = evaluate_primary_connection(
        ue_coords, bs_coords, hap_coord, leo_coord, get_all_link_budgets
    )
    
    # Debugging Block (Uncomment to inspect primary ground connections):
    # ----------------------------------------------------------------------------------------------
    debug_records = []
    for _, row in primary_df[~primary_df['Is_NTN']].iterrows():
        d_jn = row['Dist_m']
        pl_jn = path_loss(d_jn, const.CARRIER_FREQ_GHZ)
        
        debug_records.append({
            'UE_ID': row['UE_ID'],
            'Primary_RU': row['Primary_RU'],
            'd_jn_m': d_jn,
            'PL_jn_dB': round(pl_jn, 2),
            'h_sq_primary': f"{row['h_sq_jn']:.4e}",
            'Signal_W': f"{row['Signal_W']:.4e}",
            'Interf_W': f"{row['Interf_W']:.4e}",
            'Noise_W': f"{row['Noise_W']:.4e}",
            'SINR_dB': row['SINR_dB'],
            'Eps_Wireless': row['Eps_Wireless'],
            'a_jn': row['a_jn']
        })
    
    debug_df = pd.DataFrame(debug_records)
    print("\n" + "="*115)
    print("      EXTENSIVE DEBUG: SINR MATHEMATICAL COMPONENTS FOR PRIMARY GROUND CONNECTIONS")
    print("      Equation: SINR = Signal_W / (Interf_W + Noise_W)")
    print("="*115)
    print(debug_df.head(20).to_string(index=False)) 
    print("="*115 + "\n")
    # ----------------------------------------------------------------------------------------------- 
    
    # 3. Apply Multi-Connectivity Scheme
    print("Applying Multi-Connectivity Logic")
    mc_df = apply_mc_scheme(primary_df)
    
   # 4. Print Summary
    print("\n NETWORK CONNECTION SUMMARY (MC SCHEME) ")
    
    # Calculate Primary Distribution
    pri_gbs = len(mc_df[~mc_df['Is_NTN']])
    pri_ntn = len(mc_df[mc_df['Is_NTN']])
    
    # Calculate Secondary Distribution
    sec_counts = mc_df['Secondary_RU'].value_counts()
    sec_hap = sec_counts.get('HAP', 0)
    sec_leo = sec_counts.get('LEO', 0)
    sec_gbs = sum(sec_counts.get(f'GBS_{i}', 0) for i in range(const.NUM_GBS))
    
    print(f"Primary   -> GBS: {pri_gbs} UEs | NTN (HAP/LEO): {pri_ntn} UEs")
    print(f"Secondary -> GBS: {sec_gbs} UEs | HAP: {sec_hap} UEs | LEO: {sec_leo} UEs")
    print("-" * 60)
        
    # Rename columns for crystal-clear transparency
    display_df = mc_df.rename(columns={
        'SINR_dB': 'Pri_SINR_dB',
        'a_jn': 'Pri_a_jn',
        'a_j_prime_n': 'Sec_a_jn',
        'a_n': 'Final_MC_a_n'
    })
    
    cols_to_print = [
        'UE_ID', 'Primary_RU', 'Pri_SINR_dB', 'Pri_a_jn', 
        'Secondary_RU', 'Sec_SINR_dB', 'Sec_a_jn', 'Final_MC_a_n'
    ]
    
    print("\n" + display_df[cols_to_print].head(200).to_string(index=False))
    
    # 5. Visualize
    plot_topology(mc_df, bs_coords, hap_coord, leo_coord, ue_coords, title="Multi-Connectivity Path Allocation")
    

if __name__ == "__main__":
    
    # Run the primary and MC scheme evaluation
    evaluate_mc_scheme()
    print("\n" + "="*60)
    print("--- RUNNING BACKUP ALLOCATION RESILIENCE SIMULATIONS ---")
    print("="*60)
    
    # Run backup resilience simulations
    # Generate the visual proof of risk-disjoint for the case of 100 UEs
    _, _, _, single_run_composition,_,_= run_simulation(num_users=100, num_gbs=7, fixed_rb_value=10, seed_val=42, verbose=False)
    plot_risk_disjoint_proof(single_run_composition)

    # Run the main Monte Carlo simulation
    print("\n Running Full Monte Carlo Batch ")
    total_users = [50,100,200,300,400,500]

    res_10 = run_monte_carlo_averaging(total_users, num_gbs=7, fixed_rbs=10, runs_per_scenario=50)
    res_20 = run_monte_carlo_averaging(total_users, num_gbs=7, fixed_rbs=20, runs_per_scenario=50)
        
    generate_report(total_users, res_10, res_20)




