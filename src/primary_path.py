import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from src.topology import get_hexagonal_bs, get_random_users, get_ntn_nodes, draw_hexagon
from src.system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, sinr, rate,
    GAIN_GBS_DBI, GAIN_HAP_DBI, GAIN_LEO_DBI,
    K_UMA_DB_MEAN, K_UMA_DB_SD, K_HAP_STATIC, K_LEO_STATIC
)
import src.constants as const

def dbm_to_watts(dbm):
    """Converts dBm value to Watts."""
    return 10 ** ((dbm - 30) / 10)
 
# Evaluate physical links
def get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord):
    """Calculates received power for every node and returns a list of dictionaries"""
    links = []
    
    # NTN Links
    d_hap = distance_3D(hap_coord, ue_pos)
    h_sq_hap = channel_coefficient(GAIN_HAP_DBI, free_space_path_loss(d_hap, const.CARRIER_FREQ_GHZ), K_HAP_STATIC)**2
    rx_hap = const.TX_POWER_HAP_W * h_sq_hap
    links.append({
        'name': 'HAP', 'rx_w': rx_hap, 'is_ntn': True, 'pos': hap_coord, 
        'dist': d_hap, 'p_tx': const.TX_POWER_HAP_W, 'h_sq': h_sq_hap
    })
    
    d_leo = distance_3D(leo_coord, ue_pos)
    h_sq_leo = channel_coefficient(GAIN_LEO_DBI, free_space_path_loss(d_leo, const.CARRIER_FREQ_GHZ), K_LEO_STATIC)**2
    rx_leo = const.TX_POWER_LEO_W * h_sq_leo
    links.append({
        'name': 'LEO', 'rx_w': rx_leo, 'is_ntn': True, 'pos': leo_coord, 
        'dist': d_leo, 'p_tx': const.TX_POWER_LEO_W, 'h_sq': h_sq_leo
    })

    # Terrestrial Links
    gbs_powers_w = [] 
    
    for i, bs_pos in enumerate(bs_coords):
        d_gbs = distance_3D(bs_pos, ue_pos) 
        k_db = np.random.normal(K_UMA_DB_MEAN, K_UMA_DB_SD) 
        h_sq_gbs = channel_coefficient(GAIN_GBS_DBI, path_loss(d_gbs, const.CARRIER_FREQ_GHZ), k_db)**2
        rx_gbs = const.TX_POWER_GBS_W * h_sq_gbs
        
        gbs_powers_w.append(rx_gbs)  
        links.append({
            'name': f'GBS_{i}', 'rx_w': rx_gbs, 'is_ntn': False, 'pos': bs_pos, 
            'dist': d_gbs, 'p_tx': const.TX_POWER_GBS_W, 'h_sq': h_sq_gbs
        })
        
    return links, gbs_powers_w

# 3D plotting
def plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords):
    """Handles all Matplotlib rendering separately from the logic."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for bs in bs_coords: 
        draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, label='UE')
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=120, label='Ground BS')
    ax.scatter(*hap_coord, c='black', marker='^', s=120, label='HAP')
    ax.scatter(*leo_coord, c='green', marker='^', s=120, label='LEO')

    # Draw lines using stored coordinates in DataFrame
    for _, row in df.iterrows():
        u_pos = ue_coords[row['UE_Idx']]
        r_pos = row['RU_Pos']
        color = '#ff7f0e' if row['Is_NTN'] else '#1f77b4'
        ax.plot([u_pos[0], r_pos[0]], [u_pos[1], r_pos[1]], [u_pos[2], r_pos[2]], color=color, alpha=0.6, lw=1)

    ax.set_box_aspect([1, 1, 0.6])
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)', title='Primary Path Allocation')
    plt.tight_layout()
    plt.show()
    
def analyze_primary_paths():
    """Main loop for greedy allocation."""
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    
    np.random.seed(42) 
    ue_coords = get_random_users(n=const.NUM_UE)
    results = []

    for ue_id, ue_pos in enumerate(ue_coords):
        # 1. Calculate all available links for this UE
        links, gbs_powers = get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord)
        
        # 2. Find the dictionary with the highest rx_w
        best_link = max(links, key=lambda x: x['rx_w'])
        
        # 3. Calculate final metrics based on the best link
        interference_w = 0.0 if best_link['is_ntn'] else sum(gbs_powers) - best_link['rx_w']
        
        sinr_lin = sinr(1.0, best_link['rx_w'], interference_w, const.NOISE_SPECTRAL_DENSITY_W, const.BANDWIDTH_HZ)
        
        MAX_SE = 8.0 # Maximum Spectral effeciency of 8 bps/Hz considering maximum of 256-QAM
        new_sinr_lin = min(np.log2(1+sinr_lin), MAX_SE) ; # Capped to maximum of 256-QAM
        
        cap_mbps = rate(const.BANDWIDTH_HZ, new_sinr_lin) / 1e6

        # Append structured data
        results.append({
            "UE_Idx": ue_id, 
            "UE_ID": f"UE_{ue_id:02d}",
            "Primary_RU": best_link['name'],
            "Is_NTN": best_link['is_ntn'],
            "RU_Pos": best_link['pos'], 
            "Rx_Power_dBm": round(10 * np.log10(best_link['rx_w']) + 30, 2),
            "Capacity_Mbps": round(cap_mbps, 2)
        })

    # Print  Summary 
    df = pd.DataFrame(results)
    
    print("\nNETWORK CONNECTION SUMMARY ")
    print(f"Ground BS connected UEs  : {len(df[~df['Is_NTN']])}")
    print(f"HAP connected UEs        : {len(df[df['Primary_RU'] == 'HAP'])}")
    print(f"LEO connected UEs        : {len(df[df['Primary_RU'] == 'LEO'])}")
    print(f"System Cap : {df['Capacity_Mbps'].sum():.2f} Mbps\n")

    print(df[['UE_ID', 'Primary_RU', 'Rx_Power_dBm', 'Capacity_Mbps']].to_string(index=False))
    
    # Send all data to the plotting function
    plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords)

if __name__ == "__main__":
    analyze_primary_paths()
    
