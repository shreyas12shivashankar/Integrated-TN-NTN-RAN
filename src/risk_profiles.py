from src.primary_path import get_all_link_budgets

def inject_bs_failure(ue_coords, bs_coords, hap_coord, leo_coord, failed_bs_indices, evaluate_link_func):
    """ Simulates Ground Base Station failure risk """
    affected_users = []
    
    for ue_id, ue_pos in enumerate(ue_coords):
        # 1. Fetch baseline links
        links, gbs_powers = get_all_link_budgets(ue_pos, bs_coords, hap_coord, leo_coord)
        
        # 2. Determine Primary Path
        primary_link = max(links, key=lambda x: x['rx_w'])
        
        # 3. Check if their primary path is affected
        if primary_link['is_ntn'] or int(primary_link['name'].split('_')[1]) not in failed_bs_indices:
            continue
            
        # 4. User is affected. Hence their primary availability is 0
        user_links = {
            "ue_id": ue_id, 
            "primary_availability": 0.0, 
            "candidate_links": {}
        }
        
        raw_interference = sum(p for idx, p in enumerate(gbs_powers) if idx not in failed_bs_indices)

        # 5. Calculate backup candidates under the degraded network conditions
        for link in links:
            name = link['name']
            
            if not link['is_ntn']:
                bs_id = int(name.split('_')[1])
                
                # Skip failed nodes
                if bs_id in failed_bs_indices:
                    continue
            
                interference = raw_interference - link['rx_w']
            else: 
                interference = 0.0
            
            # Evaluate the degraded link
            success, a_jn = evaluate_link_func(link['p_tx'], link['h_sq'], interference,link['dist'])
            if success:
                user_links["candidate_links"][name] = a_jn
                
        affected_users.append(user_links)
        
    return affected_users

# Further risks simulation:
# def inject_weather_condition()
# def inject_low_sinr_outage()