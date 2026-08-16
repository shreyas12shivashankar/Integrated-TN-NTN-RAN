# # Message passing algorithm scheduler v4

# import numpy as np

# def run_mpa_scheduler(affected_users, active_nodes, fixed_rb_value=10, iterations=10):
#     """
#     Executes a Message Passing Algorithm to globally resolve MAC-layer contention,
#     eliminating sequential queue bias using vectorized NumPy operations.
#     """
#     N = len(affected_users)
#     M = len(active_nodes)
    
#     if N == 0 or M == 0:
#         return 0, {}, {}

#     # Map nodes to matrix indices for easy lookup
#     node_to_idx = {node: idx for idx, node in enumerate(active_nodes)}
#     idx_to_node = {idx: node for idx, node in enumerate(active_nodes)}
    
#     # 1. Initialize the Physical Availability Matrix (A)
#     # A[u, r] stores the static physical reliability of link from UE u to RU r
#     A = np.zeros((N, M))
    
#     for u_idx, user in enumerate(affected_users):
#         for node, a_backup in user["candidate_links"].items():
#             if node in node_to_idx:
#                 r_idx = node_to_idx[node]
#                 A[u_idx, r_idx] = a_backup
                
#     # 2. Initialize the State Matrix (B)
#     # At t=0, the End-to-End availability is simply the physical availability
#     B = np.copy(A)
    
#     print(f"--- Starting MPA Convergence ({iterations} Iterations) ---")
    
#     # 3. The Iterative Message Passing Loop
#     for t in range(iterations):
#         # Step A: UEs broadcast demand weights (W) to RUs
#         # W_ur = B_ur / sum(B_u)
#         row_sums = np.sum(B, axis=1, keepdims=True)
#         # Add epsilon (1e-9) to prevent division by zero for isolated UEs
#         W = B / (row_sums + 1e-9) 
        
#         # Step B: RUs calculate total received load (D)
#         D = np.sum(W, axis=0)
        
#         # Step C: RUs broadcast the global congestion penalty (phi) back to UEs
#         phi = np.minimum(1.0, fixed_rb_value / (D + 1e-9))
        
#         # Step D: UEs update their End-to-End Availability (B) for the next iteration
#         B = A * phi
        
#         # Optional: Print the average penalty across the network to watch it converge
#         print(f"Iteration {t+1:<2} | Mean Network Penalty (phi): {np.mean(phi):.4f}")

#     # 4. Hard Assignment Resolution
#     # MPA provides fair, converged probabilities. We now lock in the physical RBs
#     # by assigning the highest converged B-scores first.
    
#     global_node_usage = {node: 0 for node in active_nodes}
#     node_composition = {node: {} for node in active_nodes}
#     recovered_count = 0
#     assigned_ue_indices = set()
    
#     # Flatten the matrix and get indices sorted by highest converged E2E availability
#     flat_indices = np.argsort(B, axis=None)[::-1]
    
#     for flat_idx in flat_indices:
#         u_idx, r_idx = np.unravel_index(flat_idx, B.shape)
        
#         # Stop if the remaining best score is 0 (no viable link)
#         if B[u_idx, r_idx] == 0:
#             break
            
#         # If user is already assigned, or the node is physically full, skip
#         node = idx_to_node[r_idx]
#         if u_idx in assigned_ue_indices or global_node_usage[node] >= fixed_rb_value:
#             continue
            
#         # Lock in the assignment
#         assigned_ue_indices.add(u_idx)
#         global_node_usage[node] += 1
#         recovered_count += 1
        
#         primary_node = affected_users[u_idx].get("primary_node", "Unknown")
#         if primary_node not in node_composition[node]:
#             node_composition[node][primary_node] = 0
#         node_composition[node][primary_node] += 1

#     return recovered_count, global_node_usage, node_composition


# # ==========================================
# # SYNTHETIC TEST ENVIRONMENT
# # ==========================================
# if __name__ == "__main__":
#     # Generate 90 stranded UEs (30 from GBS_4, 30 from GBS_5, 30 from GBS_6)
#     np.random.seed(42)
#     mock_affected_users = []
    
#     failed_sources = ['GBS_4', 'GBS_5', 'GBS_6']
#     active_nodes = ['HAP', 'LEO', 'GBS_0', 'GBS_1', 'GBS_2', 'GBS_3']
    
#     for i in range(90):
#         primary = failed_sources[i // 30]
        
#         # Mock randomized physical availabilities (a_backup) for each candidate link
#         # HAP and LEO are globally available, surviving GBSs are localized
#         candidate_links = {}
#         candidate_links['HAP'] = np.random.uniform(0.7, 0.95)
#         candidate_links['LEO'] = np.random.uniform(0.6, 0.9)
        
#         # Randomly connect to 1 or 2 surviving terrestrial nodes with varied signal
#         local_gbs = np.random.choice(['GBS_0', 'GBS_1', 'GBS_2', 'GBS_3'], size=2, replace=False)
#         for gbs in local_gbs:
#             candidate_links[gbs] = np.random.uniform(0.4, 0.99)
            
#         mock_affected_users.append({
#             "ue_id": i,
#             "primary_node": primary,
#             "primary_availability": 0.0,
#             "candidate_links": candidate_links
#         })

#     # Run the Vectorized MPA Scheduler
#     print("\nExecuting Vectorized MPA Scheduler...")
#     recovered, loads, composition = run_mpa_scheduler(
#         affected_users=mock_affected_users, 
#         active_nodes=active_nodes, 
#         fixed_rb_value=10,
#         iterations=8
#     )
    
#     # Print Results
#     print("\n--- Final Network State ---")
#     print(f"Total Affected UEs : 90")
#     print(f"Total Recovered UEs: {recovered}")
#     print(f"Network Resilience : {(recovered/90)*100:.2f}%\n")
    
#     print(f"{'Node':<8} | {'Total Load':<12} | {'Risk Composition'}")
#     print("-" * 65)
#     for node in active_nodes:
#         comp = composition.get(node, {})
#         comp_str = ", ".join([f"{k}: {v}" for k, v in comp.items()])
#         print(f"{node:<8} | {loads[node]:<2} / 10 RBs | {comp_str}")

