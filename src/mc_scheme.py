import numpy as np
import pandas as pd
from src.system_model import sinr, error_probability, check_transmission_success
import src.constants as const

def apply_mc_scheme(primary_df):
    """
    Takes the primary baseline and applies a Dual-Connectivity MC scheme.
    Every user gets exactly one primary and one secondary path,
    simulating packet duplication for seamless URLLC reliability.
    Enforces strict 50 RB capacity limits on all nodes (GBS, HAP, LEO).
    """
    RHO_PHYSICAL = 1.0 
    PSI_BACKHAUL = 1.0 - const.BACKHAUL_ERROR_PROB
    
    # Initialize capacity for ALL network nodes (Total 50 RBs each)
    node_capacity = {f'GBS_{i}': const.TOTAL_RBS_GBS for i in range(const.NUM_GBS)}
    node_capacity['HAP'] = const.TOTAL_RBS_GBS
    node_capacity['LEO'] = const.TOTAL_RBS_GBS

    # Deduct capacity used by the primary connections
    for _, row in primary_df.iterrows():
        node_capacity[row['Primary_RU']] -= 1

    mc_results = []

    for _, row in primary_df.iterrows():
        ue_id = row['UE_Idx']
        a_jn = row['a_jn']
        a_n = a_jn
        
        links = row['All_Links']
        gbs_powers = row['GBS_Powers']
        primary_ru_name = row['Primary_RU']
        
        sec = None 
        remaining_cands = []
        
        # Calculate individual availability for all remaining valid links
        for cand in links:
            if cand['name'] == primary_ru_name: 
                continue
            # Skip any node that is out of capacity
            if node_capacity[cand['name']] <= 0: 
                continue
            
            c_interf = 0.0 if cand['is_ntn'] else sum(gbs_powers) - cand['rx_w']
            c_sinr_lin = sinr(p_jn=cand['p_tx'], h_sq=cand['h_sq'], interference_power=c_interf,
                              noise_density=const.NOISE_SPECTRAL_DENSITY_W, bandwidth=const.BANDWIDTH_RB)
            
            # Latency constraint check
            c_se = min(np.log2(1 + c_sinr_lin), 8.0)
            c_cap_mbps = (const.BANDWIDTH_RB * c_se) / 1e6
            is_success, d_total_ms = check_transmission_success(
                capacity_mbps=c_cap_mbps, distance_m=cand['dist']
            )
            # Reject if E2E delay is more than latency threshold of 30 ms 
            if not is_success:
                continue
            
            c_eps = error_probability(c_sinr_lin, M=const.MODULATION_M)
            c_a_jn = ((1.0 - c_eps) * RHO_PHYSICAL) * (PSI_BACKHAUL * RHO_PHYSICAL)
            
            cand['calc_a_jn'] = c_a_jn
            cand['calc_sinr_db'] = 10 * np.log10(c_sinr_lin)
            cand['latency_ms'] = d_total_ms
            remaining_cands.append(cand)
            
        # Sort remaining candidates by highest individual availability
        remaining_cands.sort(key=lambda x: x['calc_a_jn'], reverse=True)
        
        # Assign exactly one secondary path if available
        if remaining_cands:
            sec = remaining_cands[0]
            a_n = 1.0 - ((1.0 - a_jn) * (1.0 - sec['calc_a_jn']))
            node_capacity[sec['name']] -= 1 
            
        # The updated row dictionary
        row_dict = row.to_dict()
        row_dict.update({
            "Secondary_RU": sec['name'] if sec else None,
            "Sec_Is_NTN": sec['is_ntn'] if sec else None,
            "Sec_SINR_dB": round(sec['calc_sinr_db'], 2) if sec else None,
            "Sec_latency_ms": round(sec['latency_ms'], 2) if sec else None,
            "a_j_prime_n": round(sec['calc_a_jn'], 5) if sec else None,
            "Total_MC_Paths": 2 if sec else 1,
            "a_n": round(a_n, 6)
        })
        
        row_dict.pop('All_Links', None)
        row_dict.pop('GBS_Powers', None)
        
        mc_results.append(row_dict)
        
    return pd.DataFrame(mc_results)