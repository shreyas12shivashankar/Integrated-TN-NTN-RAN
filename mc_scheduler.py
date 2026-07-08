def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices, fixed_rb_value=10, verbose=False):
    """
    Executes Backup Path Allocation using a Strict Resource Block Hard Cap.
    Includes Fallback Logic: If a user's #1 node is full, they try their #2 node, etc.
    """
    if not affected_users:
        return 0, {}

    allocated_loads = {node: 0 for node in active_nodes}
    recovered_count = 0

    # Build a ranked Candidate List for every user
    user_queues = []
    for user in affected_users:
        available_links = {node: user[node] for node in active_nodes if node in user}
        
        if available_links:
            # Sort all available nodes for this specific user from Best to Worst physical link
            ranked_nodes = sorted(available_links.items(), key=lambda x: x[1], reverse=True)
            
            # Store their absolute best link score to prioritize who gets to pick first
            best_a_in = ranked_nodes[0][1]
            
            user_queues.append({
                "ue_id": user["ue_id"], 
                "ranked_nodes": ranked_nodes, # This is their fallback list
                "best_a_in": best_a_in
            })
            
    # Sort the global line so users with the strongest overall signals pick first
    user_queues.sort(key=lambda x: x["best_a_in"], reverse=True)

    # Strict Capacity Allocation with Fallback
    for user in user_queues:
        allocated = False
        
        # User loops through their ranked choices: #1, then #2, then #3 & so on
        for node, a_in in user["ranked_nodes"]:
            # If this node has RBs available, take one and stop looking
            if allocated_loads[node] < fixed_rb_value:
                allocated_loads[node] += 1
                recovered_count += 1
                allocated = True
                if verbose:
                    print(f"UE {user['ue_id']:<3} -> {node:<6} (Allocated 1 RB)")
                break 
                
        # If the loop finishes and they found zero open RBs across all nodes,it will drop
        if not allocated:
            if verbose:
                print(f"UE {user['ue_id']:<3} -> DROPPED (All preferred backup nodes are full)")

    return recovered_count, allocated_loads