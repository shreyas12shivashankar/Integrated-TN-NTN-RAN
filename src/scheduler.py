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
    if not affected_users: return 0, {}, {}, {}

    disjoint_groups = form_risk_disjoint_groups(affected_users)
    global_node_usage = {node: 0 for node in active_nodes}
    allocated_loads = {node: 0 for node in active_nodes}
    
    node_composition = {node: {} for node in active_nodes}
    
    recovered_count = 0
    final_user_scores = {} 

    for group in disjoint_groups:
        sorted_users = sorted(group, key=lambda x: x["primary_availability"])
        group_node_requests = {} 

        for user in sorted_users:
            ue_id = user["ue_id"]
            a_primary = user["primary_availability"]
            
            p_node = user.get("primary_node", "Unknown")
            
            best_node = None
            best_a_backup_raw = -1 
            
            for node, a_backup in user["candidate_links"].items():
                if node not in global_node_usage: continue
                
                # Account for users already queued for this node in the current group
                pending_users = len(group_node_requests.get(node, []))
                
                # Ensures users are dropped when RBs are exhausted
                if global_node_usage[node] + pending_users < fixed_rb_value:
                    
                    # Connect to strongest possible node by evaluating raw a_backup
                    if a_backup > best_a_backup_raw:
                        best_node = node
                        best_a_backup_raw = a_backup 
                        
            if best_node:
                if best_node not in group_node_requests:
                    group_node_requests[best_node] = []
                group_node_requests[best_node].append((ue_id, a_primary, best_a_backup_raw, p_node))

        for node, users_on_node in group_node_requests.items():
            U_i = global_node_usage[node]
            N_i = U_i + len(users_on_node)
            phi_i = min(1.0, fixed_rb_value / N_i)
            
            global_node_usage[node] += len(users_on_node) 
            allocated_loads[node] += len(users_on_node)
            
            
            for ue_id, a_primary, a_backup, p_node in users_on_node:
                recovered_count += 1
                # Calculate final score after the physical route is locked
                final_user_scores[ue_id] = a_primary + (1.0 - a_primary) * a_backup * phi_i
                
                if p_node not in node_composition[node]:
                    node_composition[node][p_node] = 0
                node_composition[node][p_node] += 1

    return recovered_count, allocated_loads, node_composition, final_user_scores