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

def allocate_backup_paths(affected_users, active_nodes, failed_bs_indices=None, fixed_rb_value=10, verbose=False):
    """Allocates users using strict per-node RB pool limits, Effective Bandwidth, and Latency bounds."""
    if not affected_users: return 0, {}, {}, {}

    # 1. Constants for Network Physics
    R_JN_HZ = 180e3             # 180 kHz per RB (LTE numerology)
    PACKET_SIZE_BITS = 32 * 8   # 256 bits payload (32 bytes)
    SPEED_OF_LIGHT = 3e8        # m/s
    
    # URLLC Constraints
    D_MAX_Q_SEC = 0.0003        # 0.3 ms max queuing delay
    EPSILON_Q = 1e-7            # 10^-7 queueing delay violation probability
    LATENCY_THRESHOLD_MS = 30.0 # Strict physical E2E deadline
    D_BACKHAUL_MS = 1.0         # Static backhaul delay (300 km assumption)
    USER_LAMBDA = 100.0         # 100 packets/s per user traffic profile

    disjoint_groups = form_risk_disjoint_groups(affected_users)
    
    # Global trackers for node resource consumption and usage
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
            best_a_jn = -1.0 
            
            for node, a_jn_raw in user["candidate_links"].items():
                if node not in global_node_usage: 
                    continue
                
                # --- BOUNDARY 1: Hard Total RB Pool Limit per Node ---
                # Each node has a finite spectrum pool (e.g., fixed_rb_value = 10 RBs total).
                # Since each user consumes 1 RB from this emergency pool per connection:
                if global_node_usage[node] + 1 > fixed_rb_value:
                    continue # Drops candidate: Node has exhausted its finite RB pool
                
                # Retrieve specific physical metrics calculated in inject_bs_failure
                se_bps_hz = user["spectral_efficiencies"][node]
                distance_m = user["distances"][node]
                
                # 2. Build the physical pipe capacity using the node's total allocated RBs
                allocated_bandwidth_hz = fixed_rb_value * R_JN_HZ
                pool_capacity_bps = se_bps_hz * allocated_bandwidth_hz
                
                # --- BOUNDARY 2: Effective Bandwidth Queue Check ---
                # Traffic scales dynamically with the number of users packed into this node's backup pool
                projected_usage = global_node_usage[node] + 1
                lambda_node = projected_usage * USER_LAMBDA
                
                # Calculate required packet processing rate to avoid violating the queue bound
                num_term = math.log(1.0 / EPSILON_Q)
                den_term = D_MAX_Q_SEC * math.log((math.log(1.0 / EPSILON_Q) / (lambda_node * D_MAX_Q_SEC)) + 1.0)
                required_ebw_pkts_sec = num_term / den_term
                
                node_service_rate_pkts_sec = pool_capacity_bps / PACKET_SIZE_BITS
                
                if node_service_rate_pkts_sec < required_ebw_pkts_sec:
                    continue # Drops candidate: Congestion violates 0.3ms queue bound
                
                # --- BOUNDARY 3: Total End-to-End Latency Check ---
                d_tx_ms = (PACKET_SIZE_BITS / pool_capacity_bps) * 1000.0
                d_prop_ms = (distance_m / SPEED_OF_LIGHT) * 1000.0
                
                d_total_ms = d_tx_ms + (D_MAX_Q_SEC * 1000.0) + d_prop_ms + D_BACKHAUL_MS
                
                if d_total_ms > LATENCY_THRESHOLD_MS:
                    continue # Drops candidate: E2E latency exceeds the 30ms physical deadline
                
                # Selection Strategy: Pick the valid candidate with the strongest reliability score
                if a_jn_raw > best_a_jn:
                    best_a_jn = a_jn_raw
                    best_node = node
                        
            if best_node:
                global_node_usage[best_node] += 1
                allocated_loads[best_node] += 1
                
                if p_node not in node_composition[best_node]:
                    node_composition[best_node][p_node] = 0
                node_composition[best_node][p_node] += 1
                
                final_user_scores[ue_id] = best_a_jn
                recovered_count += 1
            else:
                final_user_scores[ue_id] = 0.0

    return recovered_count, allocated_loads, node_composition, final_user_scores