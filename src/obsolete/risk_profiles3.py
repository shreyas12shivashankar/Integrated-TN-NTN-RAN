#===============================================================================
    # RISK PROFILE 3: Limited Radio Resources / Limited Bandwidth Outage
# ==============================================================================
def inject_limited_rb_outage(
    ue_coords,
    bs_coords,
    hap_coord,
    leo_coord,
    evaluate_link_func,
    rb_capacity,
    seed_val=None,
):#Initializes the function to simulate a limited radio resource (RB) outage scenario, where the number of available RBs at each RU is constrained, potentially leading to some UEs being denied service due to resource exhaustion. 
    #The function takes in UE coordinates, BS coordinates, NTN node coordinates, an evaluation function for link performance, the RB capacity (either as a fixed integer or a dictionary mapping RUs to their capacities), and an optional random seed for reproducibility.

    if seed_val is not None: 
        np.random.seed(seed_val)

    ru_ue_map = defaultdict(list) #Creates a dictionary that maps each RU (GBS, HAP, LEO) to a list of UEs that have it as their primary link. This will be used to track demand for each RU and identify which UEs may be affected by limited RB availability.
    ue_primary_data = {} #stores primary link information for each UE

    # 1. Determine primary RU for every UE
    for ue_id, ue_pos in enumerate(ue_coords): 
        links, gbs_powers = get_all_link_budgets(
            ue_pos, bs_coords, hap_coord, leo_coord
        ) #Fetches the link budgets for all available nodes (GBS, HAP, LEO) for the current UE position, including received powers and other relevant parameters.

        primary_link = max(links, key=lambda x: x["rx_w"])
        primary_name = primary_link["name"] # Identifies the primary link for the current UE by selecting the link with the highest received power, and retrieves its name (e.g., "GBS_0", "HAP", "LEO") for further processing.

        if primary_link["is_ntn"]:
            raw_interference = 0.0
        else:
            primary_idx = int(primary_name.split("_")[1])
            raw_interference = sum(
                g_link["rx_w"]
                for idx, g_link in enumerate(gbs_powers)
                if idx != primary_idx
            ) #Calculates the total interference affecting the primary link by summing the received powers from all other GBSs, excluding the primary GBS itself. For NTN links (HAP or LEO), interference is assumed to be zero.

        res = evaluate_link_func(
            primary_link["p_tx"],
            primary_link["h_sq"],
            raw_interference,
            primary_link["dist"],
            is_ntn=primary_link["is_ntn"],
        ) #Evaluates the performance of the primary link

        success, a_primary, psi_primary, sinr_db_primary, epsilon_primary = res[:5] #Gives the evaluation results for the primary link, including whether it meets URLLC latency requirements, its E2E availability, physical reliability, SINR in dB, and 16-QAM error probability.

        ue_primary_data[ue_id] = {
            "primary_link": primary_link,
            "links": links,
            "gbs_powers": gbs_powers,
            "a_primary": a_primary,
            "psi_primary": psi_primary,
            "sinr_db_primary": sinr_db_primary,
            "epsilon_primary": epsilon_primary,
        } #stores the primary link information and evaluation results for the current UE in a dictionary, which will be used later to assess the impact of limited RB availability on each UE.

        # Count demand for this RU
        ru_ue_map[primary_name].append(ue_id) 

    # 2. Identify resource-exhausted users and track demand/capacity metrics
    affected_user_ids = set()
    ue_resource_info = {} 

    for ru_name, requested_ue_ids in ru_ue_map.items():
        demand_N_j = len(requested_ue_ids)  #Total number of UEs requesting this RU

        # Explicit dictionary validation or integer assignment
        if isinstance(rb_capacity, dict):  #looks for a dictionary mapping RUs to their respective RB capacities. If found, it retrieves the capacity for the current RU. If the RU is not present in the dictionary, it raises a ValueError indicating that the RB capacity is missing for that RU.
            if ru_name not in rb_capacity:  
                raise ValueError(
                    f"RB capacity missing for RU: {ru_name}" #Raises an error if the current RU's name is not found in the provided rb_capacity dictionary, indicating that the RB capacity for that RU is missing.
                )
            capacity_K_j = rb_capacity[ru_name] #sets the capacity for the current RU based on the provided dictionary mapping.
        else:
            capacity_K_j = int(rb_capacity) #else, if rb_capacity is not a dictionary, it assumes that the capacity is a fixed integer value and assigns it directly to capacity_K_j.

        # Map demand/capacity for diagnostic logging
        for uid in requested_ue_ids:
            ue_resource_info[uid] = {
                "demand_N_j": demand_N_j,
                "capacity_K_j": capacity_K_j,
            } #Iterates through each UE requesting the current RU and stores the demand and capacity information in a dictionary for diagnostic logging and further analysis.

        if demand_N_j > capacity_K_j: #If the total number of UEs requesting the current RU exceeds its available RB capacity, it indicates that some UEs will be denied service due to resource exhaustion. In this case, the function proceeds to randomly select which UEs will be affected.
            shuffled_uids = list(requested_ue_ids)
            np.random.shuffle(shuffled_uids)

            denied_uids = shuffled_uids[capacity_K_j:]
            affected_user_ids.update(denied_uids)

    # 3. Generate backup candidates for denied users
    affected_users = []

    for ue_id in sorted(list(affected_user_ids)):
        u_data = ue_primary_data[ue_id]
        primary_link = u_data["primary_link"]
        primary_name = primary_link["name"]
        links = u_data["links"]
        gbs_powers = u_data["gbs_powers"]
        res_info = ue_resource_info[ue_id]

        user_links = {
            "ue_id": ue_id,
            "primary_node": primary_name,
            # Physical path is still operational
            "primary_availability": u_data["a_primary"],
            "primary_reliability": u_data["psi_primary"],
            "primary_sinr_db": u_data["sinr_db_primary"],
            "primary_error_probability": u_data["epsilon_primary"],
            # Resource, not physical, failure
            "primary_resource_available": False,
            # Capacity diagnostic metadata
            "demand_N_j": res_info["demand_N_j"],
            "capacity_K_j": res_info["capacity_K_j"],
            "risk_type": "LIMITED_RB",
            "candidate_links": {},
        } #Initializes a dictionary to store information about the affected UE, including its ID, primary node, physical link performance metrics, resource availability status, demand and capacity information for the RU, risk type, and an empty dictionary for candidate backup links.

        # Evaluate alternative paths
        for link in links:
            name = link["name"]
            if name == primary_name:
                continue

            if not link["is_ntn"]:
                bs_id = int(name.split("_")[1])
                interference = sum(
                    g_link["rx_w"]
                    for idx, g_link in enumerate(gbs_powers)
                    if idx != bs_id
                )
            else:
                #  HAP/LEO links are interference-free
                interference = 0.0

            res = evaluate_link_func(
                link["p_tx"],
                link["h_sq"],
                interference,
                link["dist"],
                is_ntn=link["is_ntn"],
            )

            cand_success, a_candidate = res[0], res[1]

            if cand_success:
                user_links["candidate_links"][name] = a_candidate

        affected_users.append(user_links)

    return affected_users