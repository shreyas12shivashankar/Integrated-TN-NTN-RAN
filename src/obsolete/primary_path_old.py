import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from topology import get_hexagonal_bs, get_random_users, get_ntn_nodes, draw_hexagon
from system_model import (
    distance_3D, path_loss, free_space_path_loss, channel_coefficient, sinr, rate, error_probability,
    GAIN_GBS_DBI, GAIN_HAP_DBI, GAIN_LEO_DBI,
    K_UMA_DB_MEAN, K_UMA_DB_SD, K_HAP_STATIC, K_LEO_STATIC
)
import constants as const

def dbm_to_watts(dbm):
    """Converts dBm value to Watts."""
    return 10 ** ((dbm - 30) / 10)
 
# Evaluate physical links
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
 
    
# 3D plotting
def plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords):
    """Handles all Matplotlib rendering separately from the logic."""
    fig = plt.figure(figsize=(12, 10)) 
    ax = fig.add_subplot(111, projection='3d')

    for bs in bs_coords: 
        draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

    ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, label='UE')
    ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=120, label='Ground BS')
    ax.scatter(*hap_coord, c='black', marker='^', s=120, label='HAP')
    ax.scatter(*leo_coord, c='green', marker='^', s=120, label='LEO')

    #  Debugging code: To see plot with named users and gbs
    #-----------------------------------------------------------------------------------
    # for i, bs in enumerate(bs_coords):
    #     # Adding a Z-offset of +200m so the text floats above the blue triangle
    #     ax.text(bs[0], bs[1], bs[2] + 200, f'GBS_{i}', fontsize=10, weight='bold', color='darkblue')

    # for i, ue in enumerate(ue_coords):
    #     # Adding a Z-offset of +100m so the text floats above the red dot
    #     ax.text(ue[0], ue[1], ue[2] + 100, f'UE_{i:02d}', fontsize=8, color='darkred')
    #------------------------------------------------------------------------------------
    
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

def analyze_primary_paths(scheme="Backup"):
    """Main loop for greedy allocation based on received power with RB capacity limits.
    Scheme : "Backup" (40 RBs in primary out of 50) or "MC" (Dual active link from two distinct RU) """
    bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
    hap_coord, leo_coord = get_ntn_nodes()
    
    np.random.seed(42) 
    ue_coords = get_random_users(n=const.NUM_UE)
    results = []
    
    # System parameters for availability calculation
    RHO_PHYSICAL = 1.0 
    PSI_BACKHAUL = 1.0 - const.BACKHAUL_ERROR_PROB
    
    # Intialize capacity trackers based on the chosen scheme
    if scheme.upper() == "Backup":
        gbs_capacity = {f'GBS_{i}': const.PRIMARY_RBS_GBS for i in range(const.NUM_GBS)}
    else: # For MC scheme
        gbs_capacity = {f'GBS_{i}': const.TOTAL_RBS_GBS for i in range(const.NUM_GBS)}
    
    # Pre-calculate all links and establish Priority Queue
    user_data_list = []
    for ue_id, ue_pos in enumerate(ue_coords):
        links, gbs_powers = get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord)
        
        # Find the max RX power from ground stations to determine user priority
        max_gbs_rx = max([l['rx_w'] for l in links if not l['is_ntn']])
        
        user_data_list.append({
            'ue_id': ue_id,
            'ue_pos': ue_pos,
            'links': links,
            'gbs_powers': gbs_powers,
            'max_gbs_rx': max_gbs_rx
        })
        
    # Sort users descending by their best ground signal 
    user_data_list.sort(key=lambda x: x['max_gbs_rx'], reverse=True)

    # Capacity-Aware Allocation Loop
    for user_data in user_data_list:
        ue_id = user_data['ue_id']
        links = user_data['links']
        gbs_powers = user_data['gbs_powers']

        # Sort all links based on received power (descending order) for this specific user
        sorted_links = sorted(links, key=lambda x: x['rx_w'], reverse=True)
        
        best_link = None
        secondary_link = None

        for cand in sorted_links:
            if cand['is_ntn']:
                if best_link is None:
                    best_link = cand
                elif secondary_link is None and scheme.upper() == "MC" :
                    secondary_link = cand
            else:
                if gbs_capacity[cand['name']] > 0:
                    if best_link is None:
                        best_link = cand
                        gbs_capacity[cand['name']] -= 1
                    elif secondary_link is None and scheme.upper() == "MC" :
                        secondary_link = cand
                        gbs_capacity[cand['name']] -=1
            
            if scheme.upper() == "Backup" and best_link is not None:
                break
            if scheme.upper() == "MC" and best_link is not None and secondary_link is not None:
                break

        # Compute metrics of primary link
        interference_w = 0.0 if best_link['is_ntn'] else sum(gbs_powers) - best_link['rx_w']
        noise_w = const.NOISE_SPECTRAL_DENSITY_W * const.BANDWIDTH_RB
        
        sinr_lin = sinr(p_jn=best_link['p_tx'], h_sq=best_link['h_sq'], interference_power=interference_w,
                        noise_density=const.NOISE_SPECTRAL_DENSITY_W, bandwidth=const.BANDWIDTH_RB)
        
        MAX_SE = 8.0 # Capped to 256-QAM (8 bps/Hz)
        spectral_efficiency = min(np.log2(1 + sinr_lin), MAX_SE)
        cap_mbps = (const.BANDWIDTH_RB * spectral_efficiency) / 1e6
        
        # Primary link wireless reliability and total E2E availability (a_jn, Eq. 10)
        eps_wireless = error_probability(sinr_lin, M=const.MODULATION_M) 
        a_jn = ((1.0 - eps_wireless) * RHO_PHYSICAL) * (PSI_BACKHAUL * RHO_PHYSICAL)

        # Compute metrics for secondary path of MC scheme
        a_j_primary_n = None
        a_n=a_jn
        sec_rx_dbm =  None
        sec_sinr_db = None
        
        if scheme.upper() == "MC" and secondary_link is not None:
            sec_interference_w = 0.0 if secondary_link['is_ntn'] else sum(gbs_powers) - secondary_link['rx_w']
            
            sec_sinr_lin = sinr(p_jn=secondary_link['p_tx'], h_sq=secondary_link['h_sq'], 
                                interference_power=sec_interference_w,
                                noise_density=const.NOISE_SPECTRAL_DENSITY_W, bandwidth=const.BANDWIDTH_RB)
            
            sec_eps_wireless = error_probability(sec_sinr_lin, M=const.MODULATION_M)
            
            # Secondary link total E2E availability
            a_j_primary_n = ((1.0 - sec_eps_wireless) * RHO_PHYSICAL) * (PSI_BACKHAUL * RHO_PHYSICAL)
            
            # Joint MC Availability Calculation
            a_n = 1.0 - ((1.0 - a_jn) * (1.0 - a_j_primary_n))
            
            sec_rx_dbm = round(10 * np.log10(secondary_link['rx_w']) + 30, 2)
            sec_sinr_db = round(10 * np.log10(sec_sinr_lin), 2)
        
        # Append structured data
        row_data = {
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
            "Capacity_Mbps": round(cap_mbps, 2)
        }
        
        if scheme.upper() == "MC":
            row_data.update({
                "Secondary_RU": secondary_link['name'] if secondary_link else None,
                "Sec_Is_NTN": secondary_link['is_ntn'] if secondary_link else None,
                "Secondary_RU_Pos": secondary_link['pos'] if secondary_link else None,
                "Sec_Rx_dBm": sec_rx_dbm,
                "Sec_SINR_dB": sec_sinr_db,
                "a_j_primary_n": round(a_j_primary_n, 5) if a_j_primary_n is not None else None,
                "a_n": round(a_n, 6)
            })
            
        results.append(row_data)

    # Sort results back by UE_Idx so the printed table stays in a logical order
    results.sort(key=lambda x: x['UE_Idx'])

    # Output Summary Table
    df = pd.DataFrame(results)

    # Debugging Block: 
    # ----------------------------------------------------------------------------------------------
    # debug_records = []
    # for _, row in df[~df['Is_NTN']].iterrows():
    #     d_jn = row['Dist_m']
    #     pl_jn = path_loss(d_jn, const.CARRIER_FREQ_GHZ)
        
    #     debug_records.append({
    #         'UE_ID': row['UE_ID'],
    #         'Primary_RU': row['Primary_RU'],
    #         'd_jn_m': d_jn,
    #         'PL_jn_dB': round(pl_jn, 2),
    #         'h_sq_primary': f"{row['h_sq_jn']:.4e}",
    #         'Signal_W': f"{row['Signal_W']:.4e}",
    #         'Interf_W': f"{row['Interf_W']:.4e}",
    #         'Noise_W': f"{row['Noise_W']:.4e}",
    #         'SINR_dB': row['SINR_dB'],
    #         'Eps_Wireless': row['Eps_Wireless'],
    #         'a_jn': row['a_jn']
    #     })
    
    # debug_df = pd.DataFrame(debug_records)
    # print("\n" + "="*115)
    # print("      EXTENSIVE DEBUG: SINR MATHEMATICAL COMPONENTS FOR PRIMARY GROUND CONNECTIONS")
    # print("      Equation: SINR = Signal_W / (Interf_W + Noise_W)")
    # print("="*115)
    # print(debug_df.head(20).to_string(index=False)) 
    # print("="*115 + "\n")
    # -----------------------------------------------------------------------------------------------

    print(f"\n-------- NETWORK CONNECTION SUMMARY ({scheme.upper()} SCHEME) ----------")
    print(f"Ground BS primary UEs    : {len(df[~df['Is_NTN']])}")
    print(f"NTN (HAP/LEO) primary UEs: {len(df[df['Is_NTN']])}")
    print(f"System Primary Capacity  : {df['Capacity_Mbps'].sum():.2f} Mbps\n")

    if scheme.upper() == "MC":
        print(df[['UE_ID', 'Primary_RU', 'Secondary_RU', 'SINR_dB', 'a_jn','Sec_SINR_dB', 'a_j_primary_n', 'a_n']].head(20).to_string(index=False))
    else:
        print(df[['UE_ID', 'Primary_RU', 'Rx_Power_dBm', 'SINR_dB', 'Eps_Wireless', 'a_jn', 'Capacity_Mbps']].head(20).to_string(index=False))
            
    # Plot topology
    plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords)

if __name__ == "__main__":
    analyze_primary_paths(scheme="MC")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from topology import get_hexagonal_bs, get_random_users, get_ntn_nodes, draw_hexagon
# from system_model import (
#     distance_3D, path_loss, free_space_path_loss, channel_coefficient, sinr, rate, error_probability,
#     GAIN_GBS_DBI, GAIN_HAP_DBI, GAIN_LEO_DBI,
#     K_UMA_DB_MEAN, K_UMA_DB_SD, K_HAP_STATIC, K_LEO_STATIC
# )
# import constants as const

# def dbm_to_watts(dbm):
#     """Converts dBm value to Watts."""
#     return 10 ** ((dbm - 30) / 10)
 
# # Evaluate physical links
# def get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord):
#     """Calculates received power and link metrics for every node using per-RB transmit power."""
#     links = []
    
#     # NTN Links (HAP & LEO)
#     d_hap = distance_3D(hap_coord, ue_pos)    
#     h_sq_hap = channel_coefficient(GAIN_HAP_DBI, free_space_path_loss(d_hap, const.CARRIER_FREQ_GHZ), K_HAP_STATIC)**2
#     rx_hap = const.TX_POWER_HAP_RB_W * h_sq_hap
#     links.append({
#         'name': 'HAP', 'rx_w': rx_hap, 'is_ntn': True, 'pos': hap_coord, 
#         'dist': d_hap, 'p_tx': const.TX_POWER_HAP_RB_W, 'h_sq': h_sq_hap
#     })
    
#     d_leo = distance_3D(leo_coord, ue_pos)
    
#     h_sq_leo = channel_coefficient(GAIN_LEO_DBI, free_space_path_loss(d_leo, const.CARRIER_FREQ_GHZ), K_LEO_STATIC)**2
#     rx_leo = const.TX_POWER_LEO_RB_W * h_sq_leo
#     links.append({
#         'name': 'LEO', 'rx_w': rx_leo, 'is_ntn': True, 'pos': leo_coord, 
#         'dist': d_leo, 'p_tx': const.TX_POWER_LEO_RB_W, 'h_sq': h_sq_leo
#     })

#     # Terrestrial Links (GBS)
#     gbs_powers_w = [] 
    
#     for i, bs_pos in enumerate(bs_coords):
#         d_gbs = distance_3D(bs_pos, ue_pos)
#         k_db = np.random.normal(K_UMA_DB_MEAN, K_UMA_DB_SD) 
#         h_sq_gbs = channel_coefficient(GAIN_GBS_DBI, path_loss(d_gbs, const.CARRIER_FREQ_GHZ), k_db)**2
#         rx_gbs = const.TX_POWER_GBS_RB_W * h_sq_gbs
        
#         gbs_powers_w.append(rx_gbs)  
#         links.append({
#             'name': f'GBS_{i}', 'rx_w': rx_gbs, 'is_ntn': False, 'pos': bs_pos, 
#             'dist': d_gbs, 'p_tx': const.TX_POWER_GBS_RB_W, 'h_sq': h_sq_gbs
#         })
        
#     return links, gbs_powers_w
 
    
# # 3D plotting
# def plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords):
#     """Handles all Matplotlib rendering separately from the logic."""
#     fig = plt.figure(figsize=(12, 10)) 
#     ax = fig.add_subplot(111, projection='3d')

#     for bs in bs_coords: 
#         draw_hexagon(ax, bs, radius=const.CELL_RADIUS)

#     ax.scatter(ue_coords[:,0], ue_coords[:,1], ue_coords[:,2], c='red', s=15, label='UE')
#     ax.scatter(bs_coords[:,0], bs_coords[:,1], bs_coords[:,2], c='blue', marker='^', s=120, label='Ground BS')
#     ax.scatter(*hap_coord, c='black', marker='^', s=120, label='HAP')
#     ax.scatter(*leo_coord, c='green', marker='^', s=120, label='LEO')

#     # Debugging code: To see plot with named users and gbs
#     # ------------------------------------------------------------------------------------
#     # for i, bs in enumerate(bs_coords):
#     #     # Adding a Z-offset of +200m so the text floats above the blue triangle
#     #     ax.text(bs[0], bs[1], bs[2] + 200, f'GBS_{i}', fontsize=10, weight='bold', color='darkblue')

#     # for i, ue in enumerate(ue_coords):
#     #     # Adding a Z-offset of +100m so the text floats above the red dot
#     #     ax.text(ue[0], ue[1], ue[2] + 100, f'UE_{i:02d}', fontsize=8, color='darkred')
#     # ------------------------------------------------------------------------------------
    
#     # Draw lines using stored coordinates in DataFrame
#     for _, row in df.iterrows():
#         u_pos = ue_coords[row['UE_Idx']]
#         r_pos = row['RU_Pos']
#         color = '#ff7f0e' if row['Is_NTN'] else '#1f77b4'
#         ax.plot([u_pos[0], r_pos[0]], [u_pos[1], r_pos[1]], [u_pos[2], r_pos[2]], color=color, alpha=0.6, lw=1)

#     ax.set_box_aspect([1, 1, 0.6])
#     ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)', title='Primary Path Allocation')
#     plt.tight_layout()
#     plt.show()

# def analyze_primary_paths(scheme="Backup"):
#     """Main loop for greedy allocation based on received power with RB capacity limits.
#     Scheme : "Backup" (40 RBs in primary out of 50) or "MC" (Dual active link from two distinct RU) """
#     bs_coords = get_hexagonal_bs(radius=const.CELL_RADIUS, num_gbs=const.NUM_GBS)  
#     hap_coord, leo_coord = get_ntn_nodes()
    
#     np.random.seed(42) 
#     ue_coords = get_random_users(n=const.NUM_UE)
#     results = []
    
#     # System parameters for availability calculation
#     RHO_PHYSICAL = 1.0 
#     PSI_BACKHAUL = 1.0 - const.BACKHAUL_ERROR_PROB
    
#     # Intialize capacity trackers based on the chosen scheme
#     if scheme.upper() == "Backup":
#         gbs_capacity = {f'GBS_{i}': const.PRIMARY_RBS_GBS for i in range(const.NUM_GBS)}
#     else: # For MC scheme
#         gbs_capacity = {f'GBS_{i}': const.TOTAL_RBS_GBS for i in range(const.NUM_GBS)}
    
#     # Pre-calculate all links and establish Priority Queue
#     user_data_list = []
#     for ue_id, ue_pos in enumerate(ue_coords):
#         links, gbs_powers = get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord)
        
#         # Find the max RX power from ground stations to determine user priority
#         max_gbs_rx = max([l['rx_w'] for l in links if not l['is_ntn']])
        
#         user_data_list.append({
#             'ue_id': ue_id,
#             'ue_pos': ue_pos,
#             'links': links,
#             'gbs_powers': gbs_powers,
#             'max_gbs_rx': max_gbs_rx
#         })
        
#     # Sort users descending by their best ground signal 
#     user_data_list.sort(key=lambda x: x['max_gbs_rx'], reverse=True)

#     # Capacity-Aware Allocation Loop
#     for user_data in user_data_list:
#         ue_id = user_data['ue_id']
#         links = user_data['links']
#         gbs_powers = user_data['gbs_powers']

#         # Sort all links based on received power (descending order) for this specific user
#         sorted_links = sorted(links, key=lambda x: x['rx_w'], reverse=True)
        
#         best_link = None
#         secondary_link = None

#         for cand in sorted_links:
#             if cand['is_ntn']:
#                 if best_link is None:
#                     best_link = cand
#                 elif secondary_link is None and scheme.upper() == "MC" :
#                     secondary_link = cand
#             else:
#                 if gbs_capacity[cand['name']] > 0:
#                     if best_link is None:
#                         best_link = cand
#                         gbs_capacity[cand['name']] -= 1
#                     elif secondary_link is None and scheme.upper() == "MC" :
#                         secondary_link = cand
#                         gbs_capacity[cand['name']] -=1
            
#             if scheme.upper() == "Backup" and best_link is not None:
#                 break
#             if scheme.upper() == "MC" and best_link is not None and secondary_link is not None:
#                 break

#         # Compute metrics of primary link
#         interference_w = 0.0 if best_link['is_ntn'] else sum(gbs_powers) - best_link['rx_w']
        
#         sinr_lin = sinr(p_jn=best_link['p_tx'], h_sq=best_link['h_sq'], interference_power=interference_w,
#                         noise_density=const.NOISE_SPECTRAL_DENSITY_W, bandwidth=const.BANDWIDTH_RB)
        
#         MAX_SE = 8.0 # Capped to 256-QAM (8 bps/Hz)
#         spectral_efficiency = min(np.log2(1 + sinr_lin), MAX_SE)
#         cap_mbps = (const.BANDWIDTH_RB * spectral_efficiency) / 1e6
        
#         # Primary link wireless reliability and total E2E availability (a_jn, Eq. 10)
#         eps_wireless = error_probability(sinr_lin, M=const.MODULATION_M) 
#         a_jn = ((1.0 - eps_wireless) * RHO_PHYSICAL) * (PSI_BACKHAUL * RHO_PHYSICAL)

#         # Append structured data
#         row_data = {
#             "UE_Idx": ue_id, 
#             "UE_ID": f"UE_{ue_id:02d}",
#             "Primary_RU": best_link['name'],
#             "Is_NTN": best_link['is_ntn'],
#             "RU_Pos": best_link['pos'], 
#             "Dist_m": round(best_link['dist'], 2),
#             "h_sq_jn": f"{best_link['h_sq']:.4e}",
#             "Rx_Power_dBm": round(10 * np.log10(best_link['rx_w']) + 30, 2),
#             "SINR_dB": round(10 * np.log10(sinr_lin), 2),
#             "Eps_Wireless": f"{eps_wireless:.2e}",
#             "a_jn": round(a_jn, 6),
#             "Capacity_Mbps": round(cap_mbps, 2)
#         }
        
#         if scheme.upper() == "MC":
#             row_data.update({
#                 "Secondary_RU": secondary_link['name'] if secondary_link else None,
#                 "Sec_Is_NTN": secondary_link['is_ntn'] if secondary_link else None,
#                 "Secondary_RU_Pos": secondary_link['pos'] if secondary_link else None
#             })
            
#         results.append(row_data)

#     # Sort results back by UE_Idx so the printed table stays in a logical order
#     results.sort(key=lambda x: x['UE_Idx'])

#     # Output Summary Table
#     df = pd.DataFrame(results)
    
#     # Debugging block to print all critical parameters value
#     #-------------------------------------------------------------------------------------------------

#     debug_records = []
#     for _, row in df[~df['Is_NTN']].iterrows():
#         d_jn = row['Dist_m']
#         pl_jn = path_loss(d_jn, const.CARRIER_FREQ_GHZ)
#         debug_records.append({
#             'UE_ID': row['UE_ID'],
#             'Primary_RU': row['Primary_RU'],
#             'd_jn_m': d_jn,
#             'PL_jn_dB': round(pl_jn, 2),
#             'h_sq_jn': row['h_sq_jn'],
#             'Rx_Power_dBm': row['Rx_Power_dBm'],
#             'SINR_dB': row['SINR_dB'],
#             'a_jn': row['a_jn']
#         })
    
#     debug_df = pd.DataFrame(debug_records)
#     print("\n" + "="*85)
#     print("      DEBUG: PRIMARY CONNECTED USERS LINK CHANNEL PARAMETERS (d_jn, PL_jn, h_jn)")
#     print("="*85)
#     print(debug_df.head(10).to_string(index=False)) # Inspect first 10 primary ground connections
#     print("="*85 + "\n")
    
#     #--------------------------------------------------------------------------------------------------
    
#     print(f"\n========== NETWORK CONNECTION SUMMARY ({scheme.upper()} SCHEME) ==========")
#     print(f"Ground BS primary UEs    : {len(df[~df['Is_NTN']])}")
#     print(f"NTN (HAP/LEO) primary UEs: {len(df[df['Is_NTN']])}")
#     print(f"System Primary Capacity  : {df['Capacity_Mbps'].sum():.2f} Mbps\n")

#     if scheme.upper() == "MC":
#         print(df[['UE_ID', 'Primary_RU', 'Secondary_RU', 'Rx_Power_dBm', 'SINR_dB', 'a_jn']].head(20).to_string(index=False))
#     else:
#         print(df[['UE_ID', 'Primary_RU', 'Rx_Power_dBm', 'SINR_dB', 'Eps_Wireless', 'a_jn', 'Capacity_Mbps']].head(20).to_string(index=False))
            
#     # Plot topology
#     plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords)
  

# if __name__ == "__main__":
#     analyze_primary_paths(scheme="Backup")
    


    
