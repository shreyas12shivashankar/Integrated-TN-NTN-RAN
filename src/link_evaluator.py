import numpy as np
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient,
    GAIN_GBS_DBI, GAIN_HAP_DBI, GAIN_LEO_DBI,
    K_UMA_DB_MEAN, K_UMA_DB_SD, K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const
 
def get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord):
    """Calculates received power and link metrics for every node using per-RB transmit power."""
    links = []
    
    # NTN Links (HAP & LEO)
    d_hap = distance_3D(hap_coord, ue_pos)
    h_sq_hap = channel_coefficient(GAIN_HAP_DBI, free_space_path_loss(d_hap, const.CARRIER_FREQ_GHZ), K_HAP_STATIC)**2
    rx_hap = const.TX_POWER_HAP_RB_W * h_sq_hap
    links.append({
        'name': 'HAP', 'rx_w': rx_hap, 'is_ntn': True, 'pos': hap_coord, 
        'dist': d_hap, 'p_tx': const.TX_POWER_HAP_RB_W, 'h_sq': h_sq_hap
    })
    
    d_leo = distance_3D(leo_coord, ue_pos)
    h_sq_leo = channel_coefficient(GAIN_LEO_DBI, free_space_path_loss(d_leo, const.CARRIER_FREQ_GHZ), K_LEO_STATIC)**2
    rx_leo = const.TX_POWER_LEO_RB_W * h_sq_leo
    links.append({
        'name': 'LEO', 'rx_w': rx_leo, 'is_ntn': True, 'pos': leo_coord, 
        'dist': d_leo, 'p_tx': const.TX_POWER_LEO_RB_W, 'h_sq': h_sq_leo
    })

    # Terrestrial Links (GBS)
    gbs_powers_w = [] 
    
    for i, bs_pos in enumerate(bs_coords):
        d_gbs = distance_3D(bs_pos, ue_pos)
        k_db = np.random.normal(K_UMA_DB_MEAN, K_UMA_DB_SD) 
        h_sq_gbs = channel_coefficient(GAIN_GBS_DBI, path_loss(d_gbs, const.CARRIER_FREQ_GHZ), k_db)**2
        rx_gbs = const.TX_POWER_GBS_RB_W * h_sq_gbs
        
        gbs_powers_w.append(rx_gbs)  
        links.append({
            'name': f'GBS_{i}', 'rx_w': rx_gbs, 'is_ntn': False, 'pos': bs_pos, 
            'dist': d_gbs, 'p_tx': const.TX_POWER_GBS_RB_W, 'h_sq': h_sq_gbs
        })
        
    return links, gbs_powers_w
