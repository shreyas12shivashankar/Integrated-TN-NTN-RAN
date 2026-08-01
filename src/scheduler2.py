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
    
    recovered_count = 0
    final_user_scores = {} 

    for group in disjoint_groups:
        N_g = len(group)
        sorted_users = sorted(group, key=lambda x: x["primary_availability"])

        for idx, user in enumerate(sorted_users):
            ue_id = user["ue_id"]
            a_primary = user["primary_availability"]
            best_node = None
            best_b_in = -1

            remaining_in_group = N_g - idx
            
            # Logic: Evaluates b_in (includes shareability) dynamically
            for node, a_backup in user["candidate_links"].items():
                if node not in global_node_usage: continue
                
                U_i = global_node_usage[node]
                
                # STRICT CAPACITY CHECK
                if U_i < fixed_rb_value:
                    N_i = U_i + remaining_in_group
                    phi_i = min(1.0, fixed_rb_value / N_i)
                    b_in = a_primary + (1.0 - a_primary) * a_backup * phi_i
                    
                    if b_in > best_b_in:
                        best_b_in = b_in
                        best_node = node
                        
            # DYNAMIC ALLOCATION: Update the usage immediately so the next user in the loop sees the new U_i
            if best_node:
                global_node_usage[best_node] += 1
                allocated_loads[best_node] += 1
                recovered_count += 1
                final_user_scores[ue_id] = best_b_in

    return recovered_count, allocated_loads, {}, final_user_scores
