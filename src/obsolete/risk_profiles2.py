import numpy as np
from src.primary_path import get_all_link_budgets


# ==============================================================================
# RISK PROFILE 1: Base Station / Node Hardware Failure
# ==============================================================================
def inject_bs_failure(
    failed_bs_id,
    ue_coords,
    bs_coords,
    hap_coord,
    leo_coord,
    evaluate_link_func,
):
    
    failed_node_name = f"GBS_{failed_bs_id}"
    affected_users = []

    for ue_id, ue_pos in enumerate(ue_coords):
        # 1. Fetch baseline links and powers
        links, gbs_powers = get_all_link_budgets(
            ue_pos, bs_coords, hap_coord, leo_coord
        )

        # 2. Identify Primary Path
        primary_link = max(links, key=lambda x: x["rx_w"])
        primary_name = primary_link["name"]

        # 3. Process only UEs whose primary link was connected to the failed node
        if primary_name == failed_node_name:
            user_links = {
                "ue_id": ue_id,
                "primary_node": primary_name,
                "primary_availability": 0.0,  # Complete outage on primary link
                "primary_reliability": 0.0,
                "primary_sinr_db": -np.inf,
                "primary_error_probability": 1.0,
                "candidate_links": {},
            }

            # 4. Evaluate backup candidates among remaining operational nodes
            for link in links:
                name = link["name"]

                # Exclude the failed node itself
                if name == failed_node_name:
                    continue

                # Calculate interference excluding the failed node from total active power
                # Sums actual received interference powers p_tx * |h|^2 from other operational GBSs
                if not link["is_ntn"]:
                    bs_id = int(name.split("_")[1])
                    interference = sum(
                        g_link["rx_w"]
                        for idx, g_link in enumerate(gbs_powers)
                        if idx != bs_id and idx != failed_bs_id
                    )
                else:
                    interference = (
                        0.0  # NTN links are interference-free in this model
                    )

                # Evaluate alternative path performance
                res = evaluate_link_func(
                    link["p_tx"],
                    link["h_sq"],
                    interference,
                    link["dist"],
                    is_ntn=link["is_ntn"],
                )

                cand_success, a_candidate = res[0], res[1]

                # Save candidate links that satisfy URLLC latency constraints
                if cand_success:
                    user_links["candidate_links"][name] = a_candidate

            affected_users.append(user_links)

    return affected_users


# ==============================================================================
# RISK PROFILE 2: Low-SINR / Channel Degradation Outage
# ==============================================================================
def inject_low_sinr_outage(
    ue_coords,
    bs_coords,
    hap_coord,
    leo_coord,
    evaluate_link_func,
    target_reliability=0.99999, #Taken From table Reliability threshold 1 − εth = 0.99999
):
    #Simulates a low-SINR outage condition where the primary path's physical
    affected_users = [] 

    for ue_id, ue_pos in enumerate(ue_coords):
        # 1. Fetch baseline links and powers
        links, gbs_powers = get_all_link_budgets(
            ue_pos, bs_coords, hap_coord, leo_coord #gets the link budgets for all nodes (GBS, HAP, LEO) for a given UE position
        )

        # 2. Identify Primary Path
        primary_link = max(links, key=lambda x: x["rx_w"]) #Selects the link with the highest received power as the primary path
        primary_name = primary_link["name"] #Determines the name of the primary link (e.g., "GBS_0", "HAP", "LEO")

        # 3. Compute co-channel interference for primary link (all GBS active)
        # Sums actual received interference powers p_tx * |h|^2 from other GBSs
        if primary_link["is_ntn"]:
            raw_interference = 0.0 #Hap and LEO links are assumed to be interference-free in this model
        else:
            primary_idx = int(primary_name.split("_")[1]) #If the primary link is a GBS, extract its index to exclude it from interference calculation
            raw_interference = sum(
                g_link["rx_w"]
                for idx, g_link in enumerate(gbs_powers)
                if idx != primary_idx
            ) #Adds up the received powers from all other GBSs to compute the total interference affecting the primary link

        # 4. Evaluate baseline primary link performance
        res = evaluate_link_func(
            primary_link["p_tx"],
            primary_link["h_sq"],
            raw_interference,
            primary_link["dist"],
            is_ntn=primary_link["is_ntn"],
        ) #Passes the primary link's parameters to the evaluation function to determine its performance metrics, including URLLC latency success, E2E availability, physical reliability, SINR in dB, and 16-QAM error probability.

        # Unpack physical channel results
        # res[0]: URLLC latency success (bool)
        # res[1]: E2E availability (a_primary = rho * psi)
        # res[2]: Physical link reliability (psi_primary = 1 - epsilon)
        # res[3]: SINR in dB
        # res[4]: 16-QAM error rate (epsilon)
        success, a_primary, psi_primary, sinr_db_primary, epsilon_primary = (
            res[0],
            res[1],
            res[2],
            res[3],
            res[4],
        ) #Returns the evaluation results for the primary link, including whether it meets URLLC latency requirements, its E2E availability, physical reliability, SINR in dB, and 16-QAM error probability.

        # 5. Model-Driven Risk Check (healthy if physical reliability meets target requirement)
        if psi_primary >= target_reliability: #Checks if the primary link's physical reliability meets or exceeds the target reliability threshold (e.g., 0.99999). If it does, the link is considered healthy, and no further action is needed for this UE.
            continue

        # 6. User is affected with a degraded primary channel
        user_links = {
            "ue_id": ue_id,
            "primary_node": primary_name,
            "primary_availability": a_primary,  # Degraded baseline score (> 0.0)
            "primary_reliability": psi_primary,  # Physical reliability (psi = 1 - epsilon)
            "primary_sinr_db": sinr_db_primary,  # Diagnostic SINR metric
            "primary_error_probability": epsilon_primary,  # 16-QAM symbol error rate
            "candidate_links": {},
        } #Initializes a dictionary to store information about the affected UE, including its ID, primary node, degraded availability score, physical reliability, SINR in dB, 16-QAM error probability, and an empty dictionary for candidate backup links.

        # 7. Evaluate dynamic backup candidate links across all alternative nodes
        for link in links: #Iterates through all available links (GBS, HAP, LEO) for the affected UE to evaluate potential backup candidates that can provide sufficient reliability and meet URLLC latency requirements.
            name = link["name"]

            if name == primary_name:
                continue  # Skip the degraded primary node itself

            if not link["is_ntn"]: # If the link is a terrestrial GBS, calculate interference from all other GBSs except the candidate itself
                bs_id = int(name.split("_")[1])
                interference = sum(
                    g_link["rx_w"]
                    for idx, g_link in enumerate(gbs_powers)
                    if idx != bs_id
                )
            else:# If the link is an NTN node (HAP or LEO), assume zero terrestrial interference
                interference = (
                    0.0  # NTN links face zero terrestrial interference
                )

            # Evaluate alternative link performance
            res = evaluate_link_func(
                link["p_tx"],
                link["h_sq"],
                interference,
                link["dist"],
                is_ntn=link["is_ntn"],
            )#Passes the candidate link's parameters to the evaluation function to determine its performance metrics, including URLLC latency success and E2E availability.
            cand_success, a_candidate = res[0], res[1]#Retrieves the evaluation results for the candidate link, including whether it meets URLLC latency requirements and its E2E availability score.

            # Save candidate links that satisfy URLLC latency constraints for scheduler allocation
            if cand_success:#If the candidate link meets URLLC latency requirements, it is added to the affected user's list of candidate links for potential backup allocation.
                user_links["candidate_links"][name] = a_candidate # Stores the candidate link's availability score

        affected_users.append(user_links) #Adds the affected user's information, including the degraded primary link and any viable backup candidates, to the overall list of affected users for further processing by the scheduler or analysis.

    return affected_users #Returns the list of affected users, each with their degraded primary link information and potential backup candidate links, for further analysis or scheduling decisions.