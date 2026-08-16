import math

def form_risk_disjoint_groups(affected_users):
    if not affected_users: return []
    risk_map = {}
    for user in affected_users:
        p_node = user.get("primary_node", "Unknown")
        if p_node not in risk_map: risk_map[p_node] = []
        risk_map[p_node].append(user)
    
    disjoint_groups = []
    while any(risk_map.values()):
        current_group = []
        for p_node in list(risk_map.keys()):
            if risk_map[p_node]: current_group.append(risk_map[p_node].pop(0))
        if current_group: disjoint_groups.append(current_group)
    return disjoint_groups

def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices=None, fixed_rb_value=10, verbose=False):
    """Allocates users using a strict Hard cap(N_i <= fixed_rb_value)."""
    if not affected_users: return 0, {}, {}, {}

    # Form the risk-disjoint groups
    disjoint_groups = form_risk_disjoint_groups(affected_users)
    
    # Trackers for the state of network
    global_node_usage = {node: 0 for node in active_nodes}
    allocated_loads = {node: 0 for node in active_nodes}
    node_composition = {node: {} for node in active_nodes}
    
    recovered_count = 0
    final_user_scores = {} 

    # Process group by group
    for group in disjoint_groups:
        
        for user in group:
            ue_id = user["ue_id"]            
            p_node = user.get("primary_node", "Unknown")
            
            best_node = None
            best_a_backup_raw = -1
            
            # Hard Cap Evaluation
            for node, a_backup_raw in user["candidate_links"].items():
                if node not in global_node_usage: continue
                
                # Only evaluate if the node has available RBs                
                if global_node_usage[node] < fixed_rb_value:
                    
                    # Connect to strongest possible node by evaluating raw a_backup
                    if a_backup_raw > best_a_backup_raw:
                        best_a_backup_raw = a_backup_raw
                        best_node = node
                        
            # Assignent and State update
            if best_node:
                global_node_usage[best_node] += 1
                allocated_loads[best_node] += 1
                
                # Shareability always evaluate to 1.0
                phi_i = min(1.0, fixed_rb_value / global_node_usage[best_node])
                final_user_scores[ue_id] = best_a_backup_raw * phi_i
                
                if p_node not in node_composition[best_node]:
                    node_composition[best_node][p_node] = 0
                node_composition[best_node][p_node] += 1
                
                recovered_count += 1
            else:
                # User is dropped due to network capacity constraint
                final_user_scores[ue_id] = 0.0

    return recovered_count, allocated_loads, node_composition, final_user_scores