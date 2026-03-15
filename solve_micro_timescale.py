import numpy as np
import cvxpy as cp
import time
import Environment

def calculate_interference_matrix(delta_v, P_v, V2V_gain_matrix, delta_i, V2I_power_W, V2I_gain_to_V2V_matrix=None, sig2=1e-14):
    n_Veh, n_RB = delta_v.shape
    Interference = np.full((n_Veh, n_RB), sig2)
    
    if delta_i is not None:
        for r in range(n_RB):
            v2i_users = np.where(delta_i[:, r] > 0.5)[0]
            if len(v2i_users) > 0:
                for v in range(n_Veh):
                    i_inter = np.sum(V2I_power_W * V2V_gain_matrix[v2i_users, v])
                    Interference[v, r] += i_inter

    P_actual = delta_v * P_v
    
    targets = [(i + 1) % n_Veh for i in range(n_Veh)]
    
    for r in range(n_RB):
        tx_powers = P_actual[:, r]
        total_rx_powers = V2V_gain_matrix.T @ tx_powers 
        for v in range(n_Veh):
            rx_idx = targets[v]
            total_p = total_rx_powers[rx_idx]
            signal_p = tx_powers[v] * V2V_gain_matrix[v, rx_idx]
            i_intra = max(0, total_p - signal_p)
            Interference[v, r] += i_intra
    return Interference

def calculate_metrics(delta_v, P_v, V2V_gain_matrix, V2V_signal_gains, 
                      delta_i, V2I_power_W, V2I_gain_to_V2V, sig2, B_v, P_c, 
                      time_factor=1.0):
    n_Veh, n_RB = delta_v.shape
    Interference = calculate_interference_matrix(delta_v, P_v, V2V_gain_matrix, delta_i, V2I_power_W, V2I_gain_to_V2V, sig2)
    P_actual = delta_v * P_v
    if V2V_signal_gains.ndim == 1:
        G_signal = np.tile(V2V_signal_gains[:, None], (1, n_RB))
    else:
        G_signal = V2V_signal_gains

    SINR = (P_actual * G_signal) / (Interference + 1e-16)
    R = np.sum(delta_v * B_v * np.log2(1 + SINR) * time_factor)
    E = np.sum(delta_v * P_v) + P_c * n_Veh 
    return R, E

import numpy as np
import cvxpy as cp

def solve_small_timescale(delta_v_init, delta_v_v2i, P_v_init, 
                          V2V_gain_matrix, V2V_signal_gains, 
                          V2I_gains_to_BS, V2I_power_W, 
                          sig2_W, B_v, P_c, gamma_0,
                          C_min_v2i, time_factor=1.0):
    n_V2V, n_RB = delta_v_init.shape
    max_outer_iters_1 = 8
    max_outer_iters_2 = 8
    epsilon = 1e-3          
    OBJ_SCALING = 1e6      
    CON_SCALING = 1e9
    omega = 1000 
    
    delta_v_val = np.array(delta_v_init)
    P_v_val = np.array(P_v_init)
    q_current = 0.0
    
    v2i_allocations = [] 
    if delta_v_v2i is not None:
        rows, cols = np.where(delta_v_v2i > 0.5)
        v2i_allocations = list(zip(rows, cols))
    
    I_allowed_v2i = np.full(n_RB, 1e9) 
    target_sinr_v2i = 2**(C_min_v2i / (B_v * time_factor)) - 1
    
    for (u_idx, r_idx) in v2i_allocations:
        g_v2i = V2I_gains_to_BS[u_idx]
        max_interf = (V2I_power_W * g_v2i) / target_sinr_v2i - sig2_W
        I_allowed_v2i[r_idx] = min(I_allowed_v2i[r_idx], max(max_interf, 1e-13))

    if V2V_signal_gains.ndim == 1:
        G_sig_mat = np.tile(V2V_signal_gains[:, None], (1, n_RB))
    else:
        G_sig_mat = V2V_signal_gains
    for out_iter in range(max_outer_iters_1):
        q_old = q_current
        current_Interference = calculate_interference_matrix(
            delta_v_val, P_v_val, V2V_gain_matrix, delta_v_v2i, V2I_power_W, None, sig2_W
        )
        
        delta_v = cp.Variable((n_V2V, n_RB), nonneg=True)
        signal_power = cp.multiply(P_v_val, G_sig_mat)
        approx_sinr = signal_power / (current_Interference + 1e-16)

        rate_expr = cp.sum(cp.multiply(delta_v, (B_v / OBJ_SCALING) * time_factor * cp.log1p(approx_sinr) / np.log(2)))
        energy_expr = cp.sum(cp.multiply(delta_v, P_v_val)) + P_c * n_V2V
        delta_k = delta_v_val
        penalty_concave = -omega * (cp.power(delta_v, 4) + cp.power(delta_v, 2))
        term_convex_approx = cp.multiply(delta_k**3, np.ones((n_V2V, n_RB))) + \
                             cp.multiply(3 * delta_k**2, (delta_v - delta_k))
        penalty_convex = 2 * omega * term_convex_approx
        penalty_term = cp.sum(penalty_concave + penalty_convex) / OBJ_SCALING

        obj_delta = cp.Maximize(rate_expr - (q_old / OBJ_SCALING) * energy_expr + penalty_term)
        
        constraints_delta = [
            delta_v <= 1,
            cp.sum(delta_v, axis=1) <= 1
        ]
        
        for r in range(n_RB):
            if I_allowed_v2i[r] < 1e8:
                interf = cp.sum(cp.multiply(delta_v[:, r], P_v_val[:, r] * V2I_gains_to_BS))
                constraints_delta.append(interf * CON_SCALING <= I_allowed_v2i[r] * CON_SCALING)
        
        constraints_delta.append(approx_sinr >= cp.multiply(delta_v, gamma_0 * 1.1))

        prob_delta = cp.Problem(obj_delta, constraints_delta)
        try:
            prob_delta.solve(solver=cp.CLARABEL)
        except:
            try:
                prob_delta.solve(solver=cp.ECOS)
            except:
                pass
                
        if delta_v.value is not None:
            delta_v_val = np.clip(delta_v.value, 0, 1)

        current_Interference_P = calculate_interference_matrix(
            delta_v_val, P_v_val, V2V_gain_matrix, delta_v_v2i, V2I_power_W, None, sig2_W
        )
        
        p_v = cp.Variable((n_V2V, n_RB), nonneg=True)
        
        snr_part = cp.multiply(p_v, G_sig_mat / (current_Interference_P + 1e-16))
        rate_expr_p = cp.sum(cp.multiply(delta_v_val, (B_v / OBJ_SCALING) * time_factor * cp.log1p(snr_part) / np.log(2)))
        energy_expr_p = cp.sum(cp.multiply(delta_v_val, p_v)) + P_c * n_V2V
        
        obj_p = cp.Maximize(rate_expr_p - (q_old / OBJ_SCALING) * energy_expr_p)
        
        constraints_p = [p_v <= 0.2] # P_max
        
        for r in range(n_RB):
            if I_allowed_v2i[r] < 1e8:
                interf = cp.sum(cp.multiply(delta_v_val[:, r], cp.multiply(p_v[:, r], V2I_gains_to_BS)))
                constraints_p.append(interf * CON_SCALING <= I_allowed_v2i[r] * CON_SCALING)
        
        constraints_p.append(snr_part >= cp.multiply(delta_v_val, gamma_0 * 1.1))
        
        prob_p = cp.Problem(obj_p, constraints_p)
        try:
            prob_p.solve(solver=cp.CLARABEL)
        except:
            try:
                prob_p.solve(solver=cp.ECOS)
            except:
                pass
        
        if p_v.value is not None:
            P_v_val = np.maximum(p_v.value, 1e-9)

        R_val, E_val = calculate_metrics(delta_v_val, P_v_val, V2V_gain_matrix, V2V_signal_gains, 
                                         delta_v_v2i, V2I_power_W, None, sig2_W, B_v, P_c, time_factor)
        if E_val > 1e-9:
            q_current = R_val / E_val
        
        if abs(q_current - q_old) < epsilon and out_iter > 3:
            break

    delta_final_bin = np.zeros_like(delta_v_val)
    for i in range(n_V2V):
        best_rb = np.argmax(delta_v_val[i, :])
        # 如果松弛解太小，可能意味着该用户无法满足约束，不分配
        if delta_v_val[i, best_rb] > 0.5: 
            delta_final_bin[i, best_rb] = 1.0

    q_final = 0.0

    P_final_val = np.copy(P_v_val)
    
    for out_iter_2 in range(max_outer_iters_2):
        q_old_2 = q_final
        
        current_Interference_2 = calculate_interference_matrix(
            delta_final_bin, P_final_val, V2V_gain_matrix, delta_v_v2i, V2I_power_W, None, sig2_W
        )
        
        p_v_2 = cp.Variable((n_V2V, n_RB), nonneg=True)
        
        snr_part_2 = cp.multiply(p_v_2, G_sig_mat / (current_Interference_2 + 1e-16))
        rate_expr_2 = cp.sum(cp.multiply(delta_final_bin, (B_v / OBJ_SCALING) * time_factor * cp.log1p(snr_part_2) / np.log(2)))
        energy_expr_2 = cp.sum(cp.multiply(delta_final_bin, p_v_2)) + P_c * n_V2V
        
        obj_2 = cp.Maximize(rate_expr_2 - (q_old_2 / OBJ_SCALING) * energy_expr_2)
        
        constraints_2 = [p_v_2 <= 0.2]
        
        for r in range(n_RB):
            if I_allowed_v2i[r] < 1e8:
                interf = cp.sum(cp.multiply(delta_final_bin[:, r], cp.multiply(p_v_2[:, r], V2I_gains_to_BS)))
                constraints_2.append(interf * CON_SCALING <= I_allowed_v2i[r] * CON_SCALING)
        
        active_links = (delta_final_bin > 0.5)
        if np.any(active_links):
             constraints_2.append(cp.multiply(delta_final_bin, snr_part_2) >= cp.multiply(delta_final_bin, gamma_0))

        prob_2 = cp.Problem(obj_2, constraints_2)
        try:
            prob_2.solve(solver=cp.CLARABEL)
        except:
             try:
                prob_2.solve(solver=cp.ECOS)
             except:
                pass

        if p_v_2.value is not None:
            P_final_val = np.maximum(p_v_2.value, 1e-9)
        R_val_2, E_val_2 = calculate_metrics(delta_final_bin, P_final_val, V2V_gain_matrix, V2V_signal_gains, 
                                             delta_v_v2i, V2I_power_W, None, sig2_W, B_v, P_c, time_factor)
        
        if E_val_2 > 1e-9:
            q_final = R_val_2 / E_val_2
        else:
            q_final = 0.0
            
        if abs(q_final - q_old_2) < epsilon and out_iter_2 > 2:
            break

    return delta_final_bin, P_final_val, q_final
