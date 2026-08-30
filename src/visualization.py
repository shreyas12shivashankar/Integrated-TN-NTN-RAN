import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from src.topology import draw_hexagon
import src.constants as const

def plot_topology(df, bs_coords, hap_coord, leo_coord, ue_coords, title='Network Topology'):
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
        
        # Plot primary path (solid line)
        r_pos = row['RU_Pos']
        color = '#ff7f0e' if row['Is_NTN'] else '#1f77b4'
        ax.plot([u_pos[0], r_pos[0]], [u_pos[1], r_pos[1]], [u_pos[2], r_pos[2]], 
                color=color, alpha=0.8, lw=1, linestyle='-')

        # Plot secondary path (dashed line)
        # Check if a secondary path was successfully assigned
        if 'Secondary_RU' in row and pd.notna(row['Secondary_RU']):
            sec_ru = row['Secondary_RU']
            sec_pos = None
            
            # Map the Node Name back to its 3D coordinates
            if sec_ru == 'HAP':
                sec_pos = hap_coord
            elif sec_ru == 'LEO':
                sec_pos = leo_coord
            elif sec_ru.startswith('GBS_'):
                gbs_idx = int(sec_ru.split('_')[1])
                sec_pos = bs_coords[gbs_idx]
            
            # Draw the secondary connection
            if sec_pos is not None:
                sec_color = '#ff7f0e' if row.get('Sec_Is_NTN', False) else '#1f77b4'
                ax.plot([u_pos[0], sec_pos[0]], [u_pos[1], sec_pos[1]], [u_pos[2], sec_pos[2]], 
                        color=sec_color, alpha=0.5, lw=1, linestyle='--')

    ax.set_box_aspect([1, 1, 0.6])
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)', title=title)
    
    # Filter duplicate labels from the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    custom_lines = [
        Line2D([0], [0], color='#1f77b4', lw=1.5, linestyle='-', label='Primary Terrestrial (GBS)'),
        Line2D([0], [0], color='#ff7f0e', lw=1.5, linestyle='-', label='Primary NTN (HAP/LEO)'),
        Line2D([0], [0], color='#1f77b4', lw=1.5, linestyle='--', alpha=0.5, label='Secondary Terrestrial (GBS)'),
        Line2D([0], [0], color='#ff7f0e', lw=1.5, linestyle='--', alpha=0.5, label='Secondary NTN (HAP/LEO)')
    ]
    
    for line in custom_lines:
        by_label[line.get_label()] = line
        
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()
