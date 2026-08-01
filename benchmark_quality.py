import numpy as np

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes
from src.risk_profiles import inject_bs_failure
from src.system_model import sinr, rate, error_probability, check_transmission_success
import src.constants as const

from src.scheduler import allocate_backup_paths as run_static_scheduler  
from src.scheduler2 import allocate_backup_paths as run_dynamic_scheduler

# def real_evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
#     snr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
#     cap_mbps = rate(const.BANDWIDTH_HZ, snr_lin) / 1e6
#     ep_wireless = error_probability(snr_lin, const.MODULATION_M)
#     a_jn = (1 - ep_wireless) * rho_wireless * (1 - const.BACKHAUL_ERROR_PROB) * rho_backhaul
#     lat_success, _ = check_transmission_success(cap_mbps, dist, 64, const.LATENCY_THRESHOLD * 1000)
#     return lat_success, a_jn

def real_evaluate_link(p_tx, h_sq, interference, dist, rho_wireless=1.0, rho_backhaul=1.0):
    snr_lin = sinr(p_tx, h_sq, interference, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
    cap_mbps = rate(const.BANDWIDTH_HZ, snr_lin) / 1e6
    ep_wireless = error_probability(snr_lin, const.MODULATION_M)
    
    # Temporary debug block
    # Initialize a counter on the function object if it doesn't exist
    if not hasattr(real_evaluate_link, "printed"):
        real_evaluate_link.printed = {"GBS": 0, "HAP": 0, "LEO": 0}
        print("\n--- Sampling ep_wireless Values ---")
        
    # Classify node by distance in meters
    node = "LEO" if dist > 110000 else "HAP" if dist > 20000 else "GBS"
    
    # Print three examples of each node type 
    if real_evaluate_link.printed[node] < 3:
        snr_db = 10 * np.log10(snr_lin) if snr_lin > 0 else -999
        print(f"[{node}] Dist: {dist/1000:>6.1f} km | SNR: {snr_db:>6.2f} dB | ep_wireless: {ep_wireless:.12f}")
        real_evaluate_link.printed[node] += 1

    # Proceed with the calculation
    a_jn = (1 - ep_wireless) * rho_wireless * (1 - const.BACKHAUL_ERROR_PROB) * rho_backhaul
    lat_success, _ = check_transmission_success(cap_mbps, dist, 64, const.LATENCY_THRESHOLD * 1000)
    
    return lat_success, a_jn

def analyze_results(affected_users, final_user_scores, allocated_loads):
    """Calculates Reliability, JFI, and Backup Resource Efficiency."""
    all_scores = []
    recovered_scores = []
    
    for user in affected_users:
        ue_id = user["ue_id"]
        score = final_user_scores.get(ue_id, 0.0) 
        all_scores.append(score)
        if score > 0:
            recovered_scores.append(score)
            
    avg_score = np.mean(all_scores) * 100
    
    # Jain's Fairness Index on Recovered Users
    if recovered_scores:
        sum_scores = np.sum(recovered_scores)
        sum_sq_scores = np.sum(np.square(recovered_scores))
        jfi = (sum_scores ** 2) / (len(recovered_scores) * sum_sq_scores) if sum_sq_scores > 0 else 0
    else:
        jfi = 0

    # Backup Resource Efficiency Calculation
    total_rbs_used = sum(allocated_loads.values())
    bre = (sum(recovered_scores) / total_rbs_used) if total_rbs_used > 0 else 0
    
    return avg_score, jfi, bre, total_rbs_used

def run_performance_metrics():
    np.random.seed(42) 
    total_users = 300 
    num_gbs = 7
    fixed_rb_value = 10
    
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=num_gbs)
    hap_coord, leo_coord = get_ntn_nodes()
    ue_coords = get_random_users(n=total_users)
    
    failed_bs_indices = [4, 5, 6] 
    active_nodes = ['HAP', 'LEO'] + [f'GBS_{i}' for i in range(num_gbs) if i not in failed_bs_indices]
    
    affected_users = inject_bs_failure(
        ue_coords, bs_coords, hap_coord, leo_coord, 
        failed_bs_indices, real_evaluate_link
    )

    # Run Schedulers
    stat_rec, stat_loads, _, stat_scores_dict = run_static_scheduler(affected_users, active_nodes, failed_bs_indices, fixed_rb_value)
    dyn_rec, dyn_loads, _, dyn_scores_dict = run_dynamic_scheduler(affected_users, active_nodes, failed_bs_indices, fixed_rb_value)
    
    print("\n Individual User Scores (Static Scheduler)")
    for ue_id, b_in_score in list(stat_scores_dict.items())[:10]: # Printing the first 10 for a clean look
        print(f"User {ue_id}: Final b_in = {b_in_score:.4f}")
    print("-----------------------------------------------\n")
    
    # Analyze Results
    stat_avg, stat_jfi, stat_bre, stat_rbs = analyze_results(affected_users, stat_scores_dict, stat_loads)
    dyn_avg, dyn_jfi, dyn_bre, dyn_rbs = analyze_results(affected_users, dyn_scores_dict, dyn_loads)
    
    label_1 = "1. Network Wide Average Reliability"
    label_2 = "2. Jain's Fairness Index (JFI)"
    label_3 = "3. Backup Resource Efficiency (BRE)"
    
    print(f"{'Resource Allocation: Fixed RB Value (10 RBs)':^85}")
    print(f"{'Performance Metric':<40} | {'Static Scheduler':<22} | {'Dynamic Scheduler'}")
    print(f"{'Total Users':<40} | {total_users:<22} | {total_users}")
    print(f"{'Total Affected Users':<40} | {len(affected_users):<22} | {len(affected_users)}")
    print(f"{'Total Recovered Users':<40} | {stat_rec:<22} | {dyn_rec}")
    print(f"{'Total RBs Consumed':<40} | {stat_rbs:<22} | {dyn_rbs}")
    print(f"{label_1:<40} | {stat_avg:.2f}%{'':<16} | {dyn_avg:.2f}%")
    print(f"{label_2:<40} | {stat_jfi:.3f}{'':<17} | {dyn_jfi:.3f}")
    print(f"{label_3:<40} | {stat_bre:.3f}{'':<17} | {dyn_bre:.3f}")
    
if __name__ == '__main__':
    run_performance_metrics()