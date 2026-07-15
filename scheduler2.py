def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value=10, verbose=False):
    """
    Executes Dynamic Risk-Aware Backup Path Allocation (Method 5).
    Recomputes shareability (phi) and E2E availability (b_in) dynamically 
    after every single user allocation to naturally enforce risk-disjoint paths.
    """
    if not affected_users:
        return 0, {}

    # Initialize tracking metrics
    allocated_loads = {node: 0 for node in active_nodes}
    recovered_count = 0

    # Step 1: Initialize remaining resources
    remaining_rbs = {node: fixed_rb_value for node in active_nodes}
    
    # Step 2: Initialize and count initial competitors for each node
    remaining_users = {node: 0 for node in active_nodes}
    for user in affected_users:
        for node in active_nodes:
            if node in user["candidate_links"]:
                remaining_users[node] += 1
                
    # Step 3: Create the dynamic pool of unallocated users
    unallocated_users = affected_users.copy()

    # Step 4: The Main Dynamic Allocation Loop
    while len(unallocated_users) > 0:
        
        # Step 5: Compute dynamic Shareability (phi) for this exact iteration
        phi = {}
        for node in active_nodes:
            N_i = remaining_users[node]
            if N_i > 0:
                # Phi dynamically shrinks as remaining RBs drop, and climbs if competitors drop faster
                phi[node] = min(1.0, remaining_rbs[node] / N_i)
            else:
                phi[node] = 1.0

        # Step 6: Compute b_in and build the queue for this iteration
        user_queues = []
        for user in unallocated_users:
            a_primary = user["primary_availability"]
            ranked_candidates = []
            
            for node in active_nodes:
                if node in user["candidate_links"]:
                    a_backup = user["candidate_links"][node]
                    
                    # If remaining_rbs hit 0, phi is 0, inherently dropping b_in to the base primary risk
                    b_in = a_primary + (1.0 - a_primary) * a_backup * phi[node]
                    ranked_candidates.append((node, b_in))
            
            if ranked_candidates:
                # Rank this specific user's candidates internally
                ranked_candidates.sort(key=lambda x: x[1], reverse=True)
                best_b_in = ranked_candidates[0][1]
                
                user_queues.append({
                    "original_dict": user, 
                    "ue_id": user["ue_id"],
                    "ranked_nodes": ranked_candidates,
                    "best_b_in": best_b_in
                })

        # Safety Check: If no users have any valid candidates left with capacity, break the loop
        if not user_queues:
            break

        # Step 7: Global Sort - Find the absolute highest priority user across the network
        user_queues.sort(key=lambda x: x["best_b_in"], reverse=True)
        
        # Step 8: Pop the highest priority user
        current_user = user_queues[0]
        user_obj = current_user["original_dict"]
        
        # Step 9: Allocate the best available node
        allocated = False
        for node, b_in in current_user["ranked_nodes"]:
            if remaining_rbs[node] > 0:
                
                # Step 10: Update Network Resources
                remaining_rbs[node] -= 1
                allocated_loads[node] += 1
                recovered_count += 1
                allocated = True
                
                if verbose:
                    print(f"UE {current_user['ue_id']:<3} -> {node:<6} (b_in: {b_in:.4f} | RBs Left: {remaining_rbs[node]})")
                break
                
        if not allocated and verbose:
            print(f"UE {current_user['ue_id']:<3} -> DROPPED (Network Exhausted)")

        # Step 11: Remove competition.
        # Regardless of whether they were allocated or dropped, this user is leaving the queue.
        # They no longer compete for ANY of their candidate nodes.
        for node in active_nodes:
            if node in user_obj["candidate_links"]:
                if remaining_users[node] > 0:
                    remaining_users[node] -= 1
                    
        # Step 12: Remove from the unallocated pool to advance the while loop
        unallocated_users.remove(user_obj)

    return recovered_count, allocated_loads