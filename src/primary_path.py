import numpy as np
import pandas as pd
from src.system_model import sinr, error_probability
import src.constants as const

def evaluate_primary_connection(ue_coords, bs_coords, hap_coord, leo_coord, link_evaluator_func):
    """
    Evaluates the primary connection for all users.
    Returns a DataFrame containing the baseline state.
    """
    RHO_PHYSICAL = 1.0 
    PSI_BACKHAUL = 1.0 - const.BACKHAUL_ERROR_PROB
    
    # Initialize capacity tracker for the primary paths
    gbs_capacity = {f'GBS_{i}': const.PRIMARY_RBS_GBS for i in range(const.NUM_GBS)}
    
    # Evaluate all physical links using the link evaluator
    user_data_list = []
    for ue_id, ue_pos in enumerate(ue_coords):
        links, gbs_powers = link_evaluator_func(ue_pos, bs_coords, hap_coord, leo_coord)
        max_gbs_rx = max([l['rx_w'] for l in links if not l['is_ntn']])
        
        user_data_list.append({
            'ue_id': ue_id,
            'ue_pos': ue_pos,
            'links': links,
            'gbs_powers': gbs_powers,
            'max_gbs_rx': max_gbs_rx
        })
        
    # Sort users descending by their best received signal 
    user_data_list.sort(key=lambda x: x['max_gbs_rx'], reverse=True)
    results = []

    # Primary Allocation Loop
    for user_data in user_data_list:
        ue_id = user_data['ue_id']
        links = user_data['links']
        gbs_powers = user_data['gbs_powers']

        # Sort candidate links by received power (Rx)
        sorted_links = sorted(links, key=lambda x: x['rx_w'], reverse=True)
        best_link = None

        for cand in sorted_links:
            if cand['is_ntn'] or gbs_capacity[cand['name']] > 0:
                best_link = cand
                if not cand['is_ntn']:
                    gbs_capacity[cand['name']] -= 1
                break

        # Compute metrics of primary link
        interference_w = 0.0 if best_link['is_ntn'] else sum(gbs_powers) - best_link['rx_w']
        noise_w = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
        
        sinr_lin = sinr(p_jn=best_link['p_tx'], h_sq=best_link['h_sq'], interference_power=interference_w,
                        noise_density=const.NOISE_SPECTRAL_DENSITY_W, bandwidth=const.BANDWIDTH_RB)
        
        MAX_SE = 8.0 
        spectral_efficiency = min(np.log2(1 + sinr_lin), MAX_SE)
        cap_mbps = (const.BANDWIDTH_RB * spectral_efficiency) / 1e6
        
        eps_wireless = error_probability(sinr_lin, M=const.MODULATION_M) 
        a_jn = ((1.0 - eps_wireless) * RHO_PHYSICAL) * (PSI_BACKHAUL * RHO_PHYSICAL)

        results.append({
            "UE_Idx": ue_id, 
            "UE_ID": f"UE_{ue_id:02d}",
            "Primary_RU": best_link['name'],
            "Is_NTN": best_link['is_ntn'],
            "RU_Pos": best_link['pos'], 
            "Dist_m": round(best_link['dist'], 2),
            "P_tx_W": best_link['p_tx'],
            "h_sq_jn": best_link['h_sq'],
            "Signal_W": best_link['rx_w'],
            "Interf_W": interference_w,
            "Noise_W": noise_w,
            "Rx_Power_dBm": round(10 * np.log10(best_link['rx_w']) + 30, 2),
            "SINR_dB": round(10 * np.log10(sinr_lin), 2),
            "Eps_Wireless": f"{eps_wireless:.2e}",
            "a_jn": round(a_jn, 5),
            "Capacity_Mbps": round(cap_mbps, 2),
            # Pass original link data so MC/Backup modules can process secondary paths
            "All_Links": links, 
            "GBS_Powers": gbs_powers
        })
        
    results.sort(key=lambda x: x['UE_Idx'])
    return pd.DataFrame(results)
