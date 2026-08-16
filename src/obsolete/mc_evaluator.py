def evaluate_multi_connectivity(link_budgets, active_nodes):
    """
    Computes Multi-Connectivity (MC) availability for each user.
    a_n = 1 - product_j (1 - a_jn)
    """
    mc_results = {}
    for ue_id, nodes_data in enumerate(link_budgets):
        unavailability_product = 1.0
        for node_id in active_nodes:
            if node_id in nodes_data:
                a_jn = nodes_data[node_id]['availability']
                unavailability_product *= (1.0 - a_jn)
        
        a_mc = 1.0 - unavailability_product
        mc_results[ue_id] = a_mc
    return mc_results