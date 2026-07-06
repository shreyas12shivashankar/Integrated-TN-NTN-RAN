
# import numpy as np

# def allocate_backup_paths(affected_users, active_nodes, rb_distribution=None):
#     """
#     Matrix-based Risk-Aware Backup Path Allocator with Strict Hard Capacity limits.
#     """
#     if rb_distribution is None:
#         rb_distribution = {node: 10 for node in active_nodes}
        
#     # Initialize the backup pool trackers
#     backup_paths = {
#         node_name: {
#             "r_i": rb_distribution.get(node_name, 0), 
#             "users": [], 
#             "failed_bs_in_path": []
#         }
#         for node_name in active_nodes
#     }
    
#     # Build the Availability Matrix (Preference Ranking)
#     user_preferences = {}
#     for user in affected_users:
#         prefs = []
#         for node in active_nodes:
#             a_in = user.get(f"a_in_{node}", 0)
#             if a_in > 0: 
#                 prefs.append((node, a_in))
                
#         # Sort highest availability first
#         prefs.sort(key=lambda x: x[1], reverse=True)
#         user_preferences[user["ue_id"]] = prefs

#     recovered_users = 0
#     dropped_users = 0

#     # Strict Capacity-Aware Allocation
#     for user in affected_users:
#         ue_id = user["ue_id"]
#         prefs = user_preferences.get(ue_id, [])
        
#         allocated = False
#         for target_node, a_in in prefs:
#             current_load = len(backup_paths[target_node]["users"])
#             max_rbs = backup_paths[target_node]["r_i"]
            
#             # STRICT HARD CAP: Only allocate if there is an empty RB available
#             if current_load < max_rbs:
#                 backup_paths[target_node]["users"].append(ue_id)
#                 backup_paths[target_node]["failed_bs_in_path"].append(user.get("primary_gbs", "Unknown"))
                
#                 # Commit the connection
#                 user["final_connection"] = f"{target_node}_Shared_Pool"
#                 user["shareability"] = 1.0  # Since 1 user takes exactly 1 RB, phi remains 1.0
#                 user["b_in"] = a_in
                
#                 allocated = True
#                 recovered_users += 1
#                 break
        
#         # If all preferred nodes were completely full, the user is immediately dropped
#         if not allocated:
#             user["final_connection"] = "Dropped"
#             user["shareability"] = 0.0
#             dropped_users += 1

#     total_affected = len(affected_users)
#     resilience_percentage = (recovered_users / total_affected) * 100 if total_affected > 0 else 100

#     return resilience_percentage, backup_paths, affected_users

def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value=10, drop_threshold=0.70, verbose=False):
    """
    Executes Risk-Aware Backup Path Allocation (Algorithm 1).
    Mathematically aligns with the global shareability formula: phi = min(1.0, B / N)
    """
    if not affected_users:
        return 0, {}

    allocated_loads = {node: 0 for node in active_nodes}
    node_demand = {node: 0 for node in active_nodes}
    user_preferences = {}
    recovered_count = 0

    # --- PASS 1: Global Demand Prediction ---
    for user in affected_users:
        available_links = {node: user[node] for node in active_nodes if node in user}
        
        if available_links:
            best_node = max(available_links, key=available_links.get)
            user_preferences[user["ue_id"]] = (best_node, available_links[best_node])
            node_demand[best_node] += 1
        else:
            user_preferences[user["ue_id"]] = (None, 0.0)

    # --- PASS 2: Global Shareability Penalty ---
    for user in affected_users:
        ue_id = user["ue_id"]
        pref_node, a_in = user_preferences[ue_id]
        
        if pref_node:
            # phi is calculated based on the TOTAL demand for this node, matching plot_availability.py
            phi = min(1.0, fixed_rb_value / node_demand[pref_node]) 
            b_in = a_in * phi 
            
            if b_in >= drop_threshold: 
                allocated_loads[pref_node] += 1
                recovered_count += 1
                if verbose:
                    print(f"UE {ue_id:<3} -> {pref_node:<6} (b_in: {b_in:.3f} | phi: {phi:.2f})")
            else:
                if verbose:
                    print(f"UE {ue_id:<3} -> DROPPED (b_in: {b_in:.3f} < {drop_threshold})")
        else:
            if verbose:
                print(f"UE {ue_id:<3} -> DROPPED (No valid physical links)")

    return recovered_count, allocated_loads