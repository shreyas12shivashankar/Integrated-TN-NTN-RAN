def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value=10, verbose=False):
    """
    Executes Risk-Aware Backup Path Allocation.
    Implements true spatial shareability by counting the actual contending users for each node.
    """
    if not affected_users:
        return 0, {}

    allocated_loads = {node: 0 for node in active_nodes} # Initially unallocated
    recovered_count = 0

    # 1. Calculate True Spatial Contention (N_i) for each active node
    competing_users = {node: 0 for node in active_nodes}
    for user in affected_users:
        for node in active_nodes:
            # If the node exists in the user dict, they successfully established a physical link in main_simulation.py
            if node in user["candidate_links"]: 
                competing_users[node] += 1
                
    # 2. Calculate Shareability (Eq. 12) dynamically based on local congestion
    shareability = {}
    for node in active_nodes:
        N_i = competing_users[node]
        if N_i > 0:
            shareability[node] = min(1.0, fixed_rb_value / N_i)
        else:
            shareability[node] = 1.0

    # 3. Compute b_in (Eq. 13) and rank candidates
    user_queues = []
    for user in affected_users:
        # Primary availability is 0 for these users since their primary path is disabled
        a_primary = user["primary_availability"]
        ranked_candidates = []
        
        for node in active_nodes:
            if node in user["candidate_links"]: 
                a_backup = user["candidate_links"][node]
                phi = shareability[node]
                
                # Compute E2E Backup Availability using the dynamic shareability weight
                b_in = a_primary + (1.0 - a_primary) * a_backup * phi
                ranked_candidates.append((node, b_in))
            
        if ranked_candidates:
            # Sort internal candidate list from best to worst b_in
            ranked_candidates.sort(key=lambda x: x[1], reverse=True)
            best_b_in = ranked_candidates[0][1]
            
            user_queues.append({
                "ue_id": user["ue_id"],
                "ranked_nodes": ranked_candidates,
                "best_b_in": best_b_in
            })
            
    # 4. Sort global line by highest b_in (Eq. 14 objective)
    user_queues.sort(key=lambda x: x["best_b_in"], reverse=True)

    # 5. Strict Allocation Loop
    for user in user_queues:
        allocated = False
        
        for node, b_in in user["ranked_nodes"]:
            if allocated_loads[node] < fixed_rb_value:
                allocated_loads[node] += 1
                recovered_count += 1
                allocated = True
                if verbose:
                    print(f"UE {user['ue_id']:<3} -> {node:<6} (b_in: {b_in:.4f} | RBs: {allocated_loads[node]}/{fixed_rb_value})")
                break 
                
        if not allocated and verbose:
            print(f"UE {user['ue_id']:<3} -> DROPPED")

    return recovered_count, allocated_loads
