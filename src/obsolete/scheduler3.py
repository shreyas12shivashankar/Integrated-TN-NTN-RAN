import math

def form_risk_disjoint_groups(affected_users):
    """Groups affected users to ensure no two users share the same primary node failure."""
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

def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices=None, fixed_rb_value=10, availability_threshold=0.75, verbose=False):
    """Allocates users using Statistical Multiplexing and a Dynamic Penalty."""
    if not affected_users: return 0, {}, {}, {}

    disjoint_groups = form_risk_disjoint_groups(affected_users)
    global_node_usage = {node: 0 for node in active_nodes}
    allocated_loads = {node: 0 for node in active_nodes}
    node_composition = {node: {} for node in active_nodes}
    
    recovered_count = 0
    final_user_scores = {} 

    for group in disjoint_groups:
        for user in group:
            ue_id = user["ue_id"]
            p_node = user.get("primary_node", "Unknown")
            
            best_node = None
            best_b_in = -1 
            
            # Dynamic argmax Evaluation with Admission Control
            for node, a_backup_raw in user["candidate_links"].items():
                if node not in global_node_usage: continue
                
                # Simulate the load addition
                temp_N_i = global_node_usage[node] + 1
                
                # Apply dynamic shareability penalty (Equation 12)
                phi_i = min(1.0, fixed_rb_value / temp_N_i)
                b_in = a_backup_raw * phi_i
                
                # ADMISSION CONTROL: Only consider this node if the penalized score survives
                if b_in >= availability_threshold:
                    if b_in > best_b_in:
                        best_b_in = b_in
                        best_node = node
                        
            # Assignment and State Update
            if best_node:
                # The user is assigned and successfully recovered
                global_node_usage[best_node] += 1
                allocated_loads[best_node] += 1
                
                if p_node not in node_composition[best_node]:
                    node_composition[best_node][p_node] = 0
                node_composition[best_node][p_node] += 1
                
                final_user_scores[ue_id] = best_b_in
                recovered_count += 1
            else:
                # The user is dropped because no node could support them above the threshold
                final_user_scores[ue_id] = 0.0
                

    return recovered_count, allocated_loads, node_composition, final_user_scores