import matplotlib.pyplot as plt
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
    for i, bs in enumerate(bs_coords):
        
        # Adding a Z-offset of +200m so the text floats above the blue triangle
        ax.text(bs[0], bs[1], bs[2] + 200, f'GBS_{i}', fontsize=10, weight='bold', color='darkblue')
    
    for i, ue in enumerate(ue_coords):
        # Adding a Z-offset of +100m so the text floats above the red dot
        ax.text(ue[0], ue[1], ue[2] + 100, f'UE_{i:02d}', fontsize=8, color='darkred')
    #------------------------------------------------------------------------------------
    
    # Draw lines using stored coordinates in DataFrame
    for _, row in df.iterrows():
        u_pos = ue_coords[row['UE_Idx']]
        r_pos = row['RU_Pos']
        color = '#ff7f0e' if row['Is_NTN'] else '#1f77b4'
        ax.plot([u_pos[0], r_pos[0]], [u_pos[1], r_pos[1]], [u_pos[2], r_pos[2]], color=color, alpha=0.6, lw=1)

    ax.set_box_aspect([1, 1, 0.6])
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)', title=title)
    plt.tight_layout()
    plt.show()
