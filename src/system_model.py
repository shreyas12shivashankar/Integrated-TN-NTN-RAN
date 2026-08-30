import numpy as np
import scipy.stats as stats
import scipy.special as sp
from scipy.special import erfc


# Antenna gains in dBi (Assuming UE with omni-directional antenna of 0 dBi)
#GAIN_GBS_DBI = 16.84    
GAIN_HAP_DBI = 32.0
GAIN_LEO_DBI = 38.0

# Ricean K-factors in dB for different links
K_UMA_DB_MEAN = 9.0   # Average K-factor for Urban Macro (UMA) terrestrial links as per 3GPP TR 38.901
K_UMA_DB_SD = 3.5
K_HAP_STATIC = 15.0   # Static K-factor for HAP links in S-band (12~15 dB)
K_LEO_STATIC = 15.0   # Static K-factor for LEO satellite links in S-band (12~15 dB)

# Channel Model Functions

def distance_3D(pos_j, pos_n):
    # Eq 1: Euclidean 3D distance between RU j and UE n.
    return np.linalg.norm(np.array(pos_j) - np.array(pos_n))

def path_loss(d_jn, fc_ghz):
    # Eq 2: Path loss in dB between Terrestrial RU j and UE n.
    return 28 + 22 * np.log10(d_jn) + 20 * np.log10(fc_ghz)

def free_space_path_loss(d_jn, fc_ghz):
    # Eq 3: Free space path loss in dB between NTN RU j and UE n.
    return 32.45 + 20 * np.log10(d_jn) + 20 * np.log10(fc_ghz)

def get_rician_fading_and_pdf(k_db):
    # Convert K from dB to linear scale
    k_lin = 10 ** (k_db / 10.0)
    
    # Calculate LoS amplitude (rho) and scattered NLoS (sigma)
    rho = np.sqrt(k_lin / (k_lin + 1))
    sigma = np.sqrt(1 / (2 * (k_lin + 1)))
    
    # Draw small scale fading magnitude |w_jn| directly using SciPy
    w_jn_mag = stats.rice.rvs(b= rho / sigma, scale=sigma)
    
    # Calculate exact PDF density (Eq 13)
    z = (w_jn_mag * rho) / (sigma**2)
    exponential_adjusted = np.exp(-((w_jn_mag - rho)**2) / (2 * sigma**2))
    bessel_scaled = sp.ive(0, z)
    pdf = (w_jn_mag / sigma**2) * exponential_adjusted * bessel_scaled
    
    return w_jn_mag, pdf

def gbs_3d_antenna_gain_db(ue_pos, bs_pos, h_gbs=25.0, h_ue=1.5, n_elements=8, tilt_deg=93.0, g_e_max_dbi=8.0):
    """
    Computes the total 3D antenna gain (in dBi) for a Ground Base Station 
    using the 3GPP TR 38.901 formulation. Automatically selects the best 
    of 3 horizontal sectors (30, 150, 270 degrees).
    """
    # 1. Calculate 2D distance and spatial offsets
    dx = ue_pos[0] - bs_pos[0]
    dy = ue_pos[1] - bs_pos[1]
    d_2d_safe = max(np.sqrt(dx**2 + dy**2), 1e-6)
    
    # 2. Zenith Angle (theta) - 0 is straight up, 90 is horizon
    delta_h = h_gbs - h_ue
    theta_deg = 90.0 + np.degrees(np.arctan(delta_h / d_2d_safe))
    
    # 3. Azimuth Angle (phi) - mapped to [0, 360)
    phi_deg = np.degrees(np.arctan2(dy, dx)) % 360.0
    
    # 4. Vertical Element Attenuation (A_V)
    # Depends only on Zenith, so it is the same for all sectors
    a_v = -min(12.0 * ((theta_deg - 90.0) / 65.0)**2, 30.0)
    
    # 5. Array Factor (g_A)
    # Depends only on Zenith, so it is the same for all sectors
    theta_rad = np.radians(theta_deg)
    tilt_rad = np.radians(tilt_deg)
    cos_diff = np.cos(theta_rad) - np.cos(tilt_rad)
    
    if abs(cos_diff) < 1e-9:
        g_a = n_elements
    else:
        num = np.sin(n_elements * np.pi * cos_diff / 2.0)**2
        den = n_elements * np.sin(np.pi * cos_diff / 2.0)**2
        g_a = num / max(den, 1e-12)
        
    # 6. Evaluate all 3 sectors to find the best horizontal match
    sectors = [30.0, 150.0, 270.0]
    best_gain_linear = 0.0
    
    for sector_azi in sectors:
        # Calculate horizontal offset wrapped to [-180, 180]
        phi_offset = (phi_deg - sector_azi + 180.0) % 360.0 - 180.0
        
        # Horizontal Element Attenuation (A_H)
        a_h = -min(12.0 * (phi_offset / 65.0)**2, 30.0)
        
        # Total 3D Element Attenuation
        a_3d = -min(-(a_v + a_h), 30.0)
        
        # Convert to linear element gain
        g_e_linear = 10.0 ** ((g_e_max_dbi + a_3d) / 10.0)
        
        # Total Linear Gain for this specific sector
        g_total_linear = g_e_linear * g_a
        
        if g_total_linear > best_gain_linear:
            best_gain_linear = g_total_linear
            
    # 7. Return the maximum gain found (converted to dBi)
    return 10.0 * np.log10(max(best_gain_linear, 1e-12))

def channel_coefficient(antenna_gain_db, path_loss_db, k_factor_db):
    # Eq 4: Channel coefficient between RU j and UE n
    # Convert dB to linear scale 
    g_jn_linear = 10 ** (antenna_gain_db / 10.0)
    path_loss_linear = 10 ** (path_loss_db / 10.0)
    
    # Both Terrestrial and NTN links use the exact Rician distribution.
    # A high K-factor (ex. 15 dB) makes the Rician fading
    # magnitude strongly concentrated around its LOS-dominated value.
    w_jn, _ = get_rician_fading_and_pdf(k_factor_db)
           
    # Return absolute channel magnitude |h_jn|
    return np.sqrt(g_jn_linear / path_loss_linear) * w_jn

def sinr(p_jn, h_sq, interference_power, noise_density, bandwidth):
    # Eq 5: SINR at UE n from RU j
    noise_power = noise_density * bandwidth
    signal_power = p_jn * h_sq
    return signal_power / (interference_power + noise_power)

def rate(bandwidth, sinr):
    # Eq 6: Achievable rate from RU j to UE n
    return bandwidth * np.log2(1 + sinr)

def error_probability(sinr, M=16):
    # Eq 7: Error probability for M-QAM modulation
    x = np.sqrt((3 * sinr * np.log2(M)) / (M - 1))
    q_function = 0.5 * erfc(x / np.sqrt(2))
    return (4 / np.log2(M)) * q_function

def check_transmission_success(capacity_mbps, distance_m, packet_size_bytes=64, max_latency_ms=30.0, error_probability=1e-5):
    """ Calculates E2E delay and checks if it meets the URLLC latency threshold. 
    Based on parameters from Salehi et al. Table I.
    """
    # 1. Convert units
    packet_size_bits = packet_size_bytes * 8
    capacity_bps = capacity_mbps * 1e6
    speed_of_light = 3e8 
    
    # 2. Calculate delay components in milliseconds
    d_trans_ms = (packet_size_bits / capacity_bps) * 1000
    d_prop_ms = (distance_m / speed_of_light) * 1000
    
    # From reference paper, queueing delay bound is 0.3 ms
    d_queue_ms = 0.3 
    
    # Backhaul/Core routing delay
    d_backhaul_ms = 1.0
    
    # ARQ re-transmission delay scaling
    eps = max(1e-12 , min(0.99, error_probability))
    
    arq_scaling_factor  = 1.0 / (1.0 - eps)
    
    d_trans_scaled_ms = d_trans_ms * arq_scaling_factor
        
    # Total one-way delay
    d_total_ms = d_trans_scaled_ms + d_prop_ms + d_queue_ms + d_backhaul_ms
    
    # 3. Evaluate Success
    is_successful = d_total_ms <= max_latency_ms
    
    return is_successful, d_total_ms