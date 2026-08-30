# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm


# # 3D PLOT: Capacity Boundary (3 RU Failures)

# def execute_3d_capacity_sweep(simulation_function):
#     print("Running 3D Capacity Boundary Sweep...")
    
#     user_range = np.arange(50, 400, 50)  # X-axis
#     rb_range = np.arange(5, 55, 5)       # Y-axis
    
#     X, Y = np.meshgrid(user_range, rb_range)
#     Z = np.zeros_like(X, dtype=float)
    
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             num_users = int(X[i, j])
#             num_rbs = int(Y[i, j])
            
#             # Force 3 failed GBS for the absolute stress test
#             returns = simulation_function(
#                 num_users=num_users, 
#                 num_gbs=7,
#                 num_failed_gbs=3,
#                 fixed_backup_rbs=num_rbs, 
#                 seed_val=42, 
#                 verbose=False
#             )
            
#             b_in_scores = [0.0]
#             for item in returns:
#                 if isinstance(item, dict) and len(item) > 0:
#                     first_val = list(item.values())[0]
#                     if isinstance(first_val, float):
#                         b_in_scores = list(item.values())
#                         break
            
#             Z[i, j] = sum(b_in_scores) / len(b_in_scores) if b_in_scores else 0.0

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k', linewidth=0.5, alpha=0.9)
    
#     ax.set_xlabel('Number of Users', fontsize=12, labelpad=10)
#     ax.set_ylabel('Reserved NTN RBs', fontsize=12, labelpad=10)
#     ax.set_zlabel('Average Availability ($b_{in}$)', fontsize=12, labelpad=10)
#     ax.set_title('TN-NTN Capacity Boundary (3 GBS Failed)', fontsize=14, pad=15)
    
#     # Constrain Z-axis from 0.5 to 1.0 to highlight the degradation slope
#     ax.set_zlim(0.0, 1.0)
#     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Average availability($b_{in})')
#     ax.view_init(elev=25, azim=-135)
    
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

def execute_design_contour_sweep(simulation_function):
    print("Running System Design Contour Sweep")
    
    # Generate the grid (same as 3D surface)
    user_range = np.arange(70, 360, 40)
    rb_range = np.arange(5, 55, 5)
    
    X, Y = np.meshgrid(user_range, rb_range)
    Z = np.zeros_like(X, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Force 3 failed GBS for the stress test scenario
            returns = simulation_function(
                num_users=int(X[i, j]), 
                num_gbs=7,
                num_failed_gbs=3,
                fixed_backup_rbs=int(Y[i, j]), 
                seed_val=42, 
                verbose=False
            )
            
            # Dynamically extract final_user_scores
            b_in_scores = [0.0]
            for item in returns:
                if isinstance(item, dict) and len(item) > 0:
                    first_val = list(item.values())[0]
                    if isinstance(first_val, float):
                        b_in_scores = list(item.values())
                        break
            
            Z[i, j] = sum(b_in_scores) / len(b_in_scores) if b_in_scores else 0.0

    # Render the Contour Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Draw the filled background heatmap (Viridis colormap)
    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1.0, 21), cmap='viridis', alpha=0.9)
    cbar = fig.colorbar(cf, ax=ax, label='Average E2E Availability ($b_{in}$)')
    
    # 2. Draw the critical Design Boundary Lines (e.g., 0.80, 0.90, 0.95)
    target_thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
    contours = ax.contour(X, Y, Z, levels=target_thresholds, colors=['black', 'black', 'red', 'orange', 'white'], linewidths=2.5)
    
    # Label the contour lines directly on the graph
    ax.clabel(contours, inline=True, fontsize=12, fmt='%1.2f')

    ax.set_xlabel('Total Users in Network', fontsize=12, weight='bold')
    ax.set_ylabel('Reserved NTN RBs (HAP/LEO)', fontsize=12, weight='bold')
    ax.set_title('O-RAN Backup Dimensioning Map (3 GBS Failed)', fontsize=14, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()