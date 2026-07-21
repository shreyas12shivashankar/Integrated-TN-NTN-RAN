def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value=10, verbose=False):
    """
    Executes Dynamic Risk-Aware Backup Path Allocation (Method 5).
    Optimized for batch execution by removing redundant loops and sorts.
    """
    if not affected_users:
        return 0, {}

    # Initialize the trackers
    allocated_loads = {node: 0 for node in active_nodes}
    recovered_count = 0
    
    # Remaining RBs at each active node
    remaining_rbs = {node: fixed_rb_value for node in active_nodes}
    
    # Number of remaining users competing for each node 
    remaining_users = {node: 0 for node in active_nodes}
    
    for user in affected_users:
        for node in user["candidate_links"]:
            remaining_users[node] += 1
            
    unallocated_users = {user["ue_id"]: user for user in affected_users}
    
    # Initial Shareability (phi) 
    phi = {}
    for node in active_nodes:
        N_i = remaining_users[node]
        phi[node] = min(1.0, remaining_rbs[node] / N_i) if N_i > 0 else 1.0

    # Main Dynamic Allocation Loop
    while unallocated_users:
        user_queues = []
        
        # Compute b_in for every remaining user
        for ue_id, user in unallocated_users.items():
            a_primary = user["primary_availability"]
            ranked_candidates = []
            
            for node, a_backup in user["candidate_links"].items():
                b_in = a_primary + (1.0 - a_primary) * a_backup * phi[node]
                ranked_candidates.append((node, b_in))
            
            if ranked_candidates:
                # Rank this specific user's candidates internally
                ranked_candidates.sort(key=lambda x: x[1], reverse=True)
                
                user_queues.append({
                    "ue_id": ue_id,
                    "user_obj": user,
                    "ranked_nodes": ranked_candidates,
                    "best_b_in": ranked_candidates[0][1]
                })

        # Safety Check
        if not user_queues:
            break

        # Find the absolute highest priority user
        current_user = max(user_queues, key=lambda x: x["best_b_in"])
        ue_id = current_user["ue_id"]
        user_obj = current_user["user_obj"]
        
        # Allocate the best available node
        allocated = False
        for node, b_in in current_user["ranked_nodes"]:
            if remaining_rbs[node] > 0:
                
                # Update Network Resources
                remaining_rbs[node] -= 1
                allocated_loads[node] += 1
                recovered_count += 1
                allocated = True
                
                if verbose:
                    print(f"UE {ue_id:<3} -> {node:<6} (b_in: {b_in:.4f} | RBs Left: {remaining_rbs[node]})")
                break
                
        if not allocated and verbose:
            print(f"UE {ue_id:<3} -> DROPPED (Network Exhausted)")

        # Remove current user from the competition and update phi
        for node in user_obj["candidate_links"]:
            if remaining_users[node] > 0:
                remaining_users[node] -= 1
                
            N_i = remaining_users[node]
            phi[node] = min(1.0, remaining_rbs[node] / N_i) if N_i > 0 else 1.0
                    
        del unallocated_users[ue_id]

    return recovered_count, allocated_loads


# Implment overall sum of a_in  for both versions of schedueler and compare
# Risk scenarions for NTN links too
# Eval benchmarks and performace metrics with comparsion for scheduler algo 