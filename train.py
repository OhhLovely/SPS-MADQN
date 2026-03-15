import numpy as np
import pandas as pd
import Environment
from solve_small_timescale import solve_small_timescale, calculate_metrics, calculate_interference_matrix
from PMADQN_Agent import PMADQN_Agent
import matplotlib.pyplot as plt
import os
import shutil

n_Veh = 40
n_RB = 18
n_V2I = 40
num_episodes = 500 
K_large_steps = 100     
T_small_steps = 3     

CHECKPOINT_DIR = "PMADQN2_checkpoints"
METRICS_FILE = "PMADQN2_training_metrics.csv"
STEP_METRICS_FILE = "PMADQN2_step_metrics.csv"
SAVE_FREQ = 10

width, height = 500, 500
down_lane, up_lane = [240], [260]
left_lane, right_lane = [260], [240]

V2I_power_dB = 23
V2I_power_W = 10**((V2I_power_dB - 30)/10)
sig2_dB = -114
sig2_W = 10**(sig2_dB / 10)
B_v = 1e6
P_c = 0.1
gamma_0_dB = 3
gamma_0 = 10**(gamma_0_dB/10) 
C_min_v2i = 0.1 * 1e6
I_pu_threshold = 2e-8
time_factor = (10 - 1) / 10

def calculate_reward(EE_t, sinr_v2v, delta_v2v, I_diff):
    norm_EE = EE_t / 5e4
    r_ee = 1.0 * norm_EE  

    v2v_violation = (sinr_v2v < gamma_0) & (delta_v2v > 0.5)
    v2v_violation_rate = np.sum(v2v_violation) / (np.sum(delta_v2v) + 1e-9)

    r_constraints = 0
    
    if v2v_violation_rate > 0:
        r_constraints -= 1 * v2v_violation_rate

    r_interf = 0
    if I_diff < 0: 
        r_interf = -1


    total_reward = r_ee + r_constraints + r_interf
    return total_reward

def main():
    env = Environment.Environ(down_lane, up_lane, left_lane, right_lane, width, height, n_veh=n_Veh)
    env.add_new_vehicles_by_number(int(n_Veh/4))
    
    # 2. Initialize Agents
    state_dim = 2 * n_RB + T_small_steps
    action_dim = n_RB + 1  
    
    agents = [PMADQN_Agent(i, state_dim, action_dim) for i in range(n_V2I)]
    
    start_episode = 0
    
    history = {
        'episode': [],
        'reward': [],
        'ee': [],
        'v2i_satisfaction': [], 
        'v2v_satisfaction': []
    }

    if not os.path.exists(STEP_METRICS_FILE):
        step_df = pd.DataFrame(columns=["episode", "step", "avg_reward_10steps", "avg_ee_10steps"])
        step_df.to_csv(STEP_METRICS_FILE, index=False)

    if os.path.exists(METRICS_FILE):
        print(f"Found {METRICS_FILE}, loading history...")
        df = pd.read_csv(METRICS_FILE)
        history = df.to_dict('list')
        if len(history['episode']) > 0:
            start_episode = history['episode'][-1]
            print(f"Resuming from Episode {start_episode + 1}")
    
    if os.path.exists(CHECKPOINT_DIR):
        print("Loading Agent Checkpoints...")
        loaded_count = 0
        for agent in agents:
            if agent.load_checkpoint(CHECKPOINT_DIR):
                loaded_count += 1
        print(f"Loaded models for {loaded_count}/{n_V2I} agents.")
    else:
        print("No checkpoint directory found. Starting fresh training.")

    delta_v_v2v_ch = np.zeros((n_Veh, n_RB))
    for row in range(n_Veh):
        while np.random.rand() < 0.5:
            col = np.random.randint(0, n_RB)
            if np.sum(delta_v_v2v_ch[:, col]) != 0 or row == col:
                continue
            delta_v_v2v_ch[row, col] = 1
    P_v_v2v_ch = np.full((n_Veh, n_RB), 0.1) 
    
    last_k_v2i_gains = np.zeros((n_V2I, n_RB)) 
    last_k_v2v_interf = np.zeros((n_V2I, n_RB)) 
    last_k_rates = np.zeros((n_V2I, T_small_steps)) 

    for episode in range(start_episode, num_episodes):
        print(f"--- Episode {episode + 1}/{num_episodes} ---")
        for i in range(5000):
            env.renew_positions()
        env.renew_channels()
        
        ep_reward_accum = 0
        ep_ee_accum = 0
        ep_v2i_sat_accum = 0
        ep_v2v_sat_accum = 0
        total_steps = 0
        
        block_reward_accum = 0
        block_ee_accum = 0
        
        initial_s = np.zeros(state_dim)
        for agent in agents:
            agent.reset_state_buffer(initial_s)
        prev_actions = [None] * n_V2I
        
        for k in range(K_large_steps):
            states_k = []
            actions_k = [] 
            delta_v_v2i_matrix = np.zeros((n_Veh, n_RB))
            channel_mask = np.ones(n_RB + 1)
            
            for idx, agent in enumerate(agents):
                # Feature Construction
                if k == 0 and episode == 0:
                    feat_gain = np.zeros(n_RB)
                    feat_interf = np.zeros(n_RB)
                else:
                    feat_gain = last_k_v2i_gains[idx] 
                    feat_interf = last_k_v2v_interf[idx]

                feat_rates = last_k_rates[idx] 

                feat_gain_log = 10 * np.log10(feat_gain + 1e-16)
                feat_gain_norm = (feat_gain_log + 130) / 60.0 
                feat_gain_norm = np.clip(feat_gain_norm, 0, 1)

                feat_interf_log = 10 * np.log10(feat_interf + 1e-16)
                feat_interf_norm = (feat_interf_log + 114) / 60.0
                feat_interf_norm = np.clip(feat_interf_norm, 0, 1)

                feat_rates_norm = feat_rates / 2e7 
                feat_rates_norm = np.clip(feat_rates_norm, 0, 1)

                s_k = np.concatenate([feat_gain_norm, feat_interf_norm, feat_rates_norm])
                states_k.append(s_k)
                
                channel_mask = np.ones(n_RB + 1)
                for r in range(n_RB):
                    estimated_impact = feat_interf[r] + V2I_power_W * feat_gain[r]
                    if estimated_impact > I_pu_threshold * 1.5: 
                        channel_mask[r] = 0.0
                channel_mask[n_RB] = 1.0
                
                action = agent.select_action(s_k, prev_actions[idx], available_channels_mask=channel_mask)
                actions_k.append(action)
                
                if action < n_RB:
                    delta_v_v2i_matrix[idx, action] = 1.0

            prev_actions = actions_k[:]

            temp_gains_sum = np.zeros((n_V2I, n_RB))
            temp_interf_sum = np.zeros((n_V2I, n_RB))
            k_c_v2i_accum = np.zeros(n_V2I) 
            
            k_r_sum = 0
            k_e_sum = 0
            k_v2v_sat_count = 0 
            k_reward_accum = 0
            
            for t in range(T_small_steps):
                env.renew_positions()
                V2V_gain_matrix, V2I_gain_vector = env.renew_channels()

                V2V_signal_gains = np.zeros((n_Veh, n_RB))
                for v in range(n_Veh):
                    target = (v + 1) % n_Veh
                    V2V_signal_gains[v, :] = V2V_gain_matrix[v, target]

                delta_opt, P_opt, _ = solve_small_timescale(
                    delta_v_v2v_ch, delta_v_v2i_matrix, P_v_v2v_ch,
                    V2V_gain_matrix, V2V_signal_gains, V2I_gain_vector,
                    V2I_power_W, sig2_W, B_v, P_c, gamma_0, C_min_v2i,
                    time_factor
                )
                delta_v_v2v_ch = delta_opt
                P_v_v2v_ch = P_opt
                
                R_val, E_val = calculate_metrics(
                    delta_opt, P_opt, V2V_gain_matrix, V2V_signal_gains,
                    delta_v_v2i_matrix, V2I_power_W, None, sig2_W, B_v, P_c, 
                    time_factor
                )
                k_r_sum += R_val
                k_e_sum += E_val
                current_ee = R_val / (E_val + 1e-9)

                current_c_v2i = np.zeros(n_V2I)
                delta_v2i_vec = np.zeros(n_V2I) 
                
                current_gains = np.tile(V2I_gain_vector[:, None], (1, n_RB))
                temp_gains_sum += current_gains
                
                interf_at_bs_per_rb = np.sum(P_opt * np.tile(V2I_gain_vector[:,None], (1, n_RB)), axis=0)
                current_interf_matrix = np.tile(interf_at_bs_per_rb, (n_V2I, 1))
                temp_interf_sum += current_interf_matrix
                
                for u_idx in range(n_V2I):
                    rb = actions_k[u_idx]
                    if rb < n_RB:
                        delta_v2i_vec[u_idx] = 1.0 
                        interf_val = interf_at_bs_per_rb[rb]
                        sinr_v2i = (V2I_power_W * V2I_gain_vector[u_idx]) / (sig2_W + interf_val + 1e-12)
                        cap = B_v * np.log2(1 + sinr_v2i) * time_factor
                        current_c_v2i[u_idx] = cap
                        k_c_v2i_accum[u_idx] += cap
                    else:
                        delta_v2i_vec[u_idx] = 0.0
                        current_c_v2i[u_idx] = 0.0
                
                last_k_rates[:, t] = current_c_v2i

                Interference = calculate_interference_matrix(
                    delta_opt, P_opt, V2V_gain_matrix, 
                    delta_v_v2i_matrix, V2I_power_W, None, sig2_W
                )
                P_actual = delta_opt * P_opt
                sinr_v2v_mat = (P_actual * V2V_signal_gains) / (Interference + 1e-16)
                
                i_su_total = np.sum(P_opt * np.tile(V2I_gain_vector[:,None], (1, n_RB))) + \
                             np.sum(V2I_power_W * V2I_gain_vector)
                i_diff = I_pu_threshold - i_su_total

                step_reward = calculate_reward(
                    current_ee, sinr_v2v_mat, delta_opt, i_diff
                )
                k_reward_accum += step_reward
                
                active_links = np.sum(delta_opt)
                if active_links > 0:
                    sat_links = np.sum((sinr_v2v_mat >= gamma_0) * delta_opt)
                    k_v2v_sat_count += (sat_links / active_links)

            last_k_v2i_gains = temp_gains_sum / T_small_steps
            last_k_v2v_interf = temp_interf_sum / T_small_steps

            avg_reward = k_reward_accum / T_small_steps
            avg_ee = k_r_sum / k_e_sum if k_e_sum > 0 else 0
            avg_c_v2i = k_c_v2i_accum / T_small_steps
            
            delta_v2i_vec_k = np.array([1.0 if a < n_RB else 0.0 for a in actions_k])
            v2i_compliant = (avg_c_v2i >= (delta_v2i_vec_k * C_min_v2i))
            v2i_sat_ratio_k = np.mean(v2i_compliant)

            avg_v2v_sat = k_v2v_sat_count / T_small_steps
            
            for idx, agent in enumerate(agents):
                s_next_constructed = np.concatenate([last_k_v2i_gains[idx], last_k_v2v_interf[idx], last_k_rates[idx]])
                done = (k == K_large_steps - 1)
                agent.store_transition(states_k[idx], actions_k[idx], avg_reward, s_next_constructed, done)
                agent.train()
            
            block_reward_accum += avg_reward
            block_ee_accum += avg_ee
            
            if (k + 1) % 10 == 0:
                block_avg_reward = block_reward_accum / 10.0
                block_avg_ee = block_ee_accum / 10.0
                
                
                new_row = {
                    "episode": episode + 1,
                    "step": k + 1,
                    "avg_reward_10steps": block_avg_reward,
                    "avg_ee_10steps": block_avg_ee
                }
                pd.DataFrame([new_row]).to_csv(STEP_METRICS_FILE, mode='a', header=False, index=False)
                
                block_reward_accum = 0
                block_ee_accum = 0

            ep_reward_accum += avg_reward
            ep_ee_accum += avg_ee
            ep_v2i_sat_accum += v2i_sat_ratio_k
            ep_v2v_sat_accum += avg_v2v_sat
            total_steps += 1
            
            if k % 10 == 0:
                print(f"Episode {episode+1} Step {k}: Reward {avg_reward:.2f}")

        history['episode'].append(episode + 1)
        history['reward'].append(ep_reward_accum / total_steps)
        history['ee'].append(ep_ee_accum / total_steps)
        
        print(f"PMADQN2 Episode {episode+1} Avg Reward: {history['reward'][-1]:.2f}")
        print(f"EE: {history['ee'][-1]:.4f}")
        print("-" * 30)
        
        if (episode + 1) % SAVE_FREQ == 0:
            print(f"Saving checkpoints for Episode {episode + 1}...")
            df = pd.DataFrame(history)
            df.to_csv(METRICS_FILE, index=False)
            
            for agent in agents:
                agent.save_checkpoint(CHECKPOINT_DIR)
            print("Checkpoints saved.")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'p-MADQN Performance (Ep {episode+1})', fontsize=16)

            axes[0].plot(history['episode'], history['reward'], 'b-')
            axes[0].set_title('Average Reward')
            axes[1].plot(history['episode'], history['ee'], 'g-')
            axes[1].set_title('Energy Efficiency')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'PMADQN2_training_analysis.pdf', dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    main()
