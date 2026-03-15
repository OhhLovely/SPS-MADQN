import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from PMADQN_Model import PMADQN_Net

class PMADQN_Agent:
    def __init__(self, agent_id, state_dim, action_dim, seq_len=5):
        self.id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.seq_len = seq_len 
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        self.q_net = PMADQN_Net(state_dim, action_dim).to(self.device)
        self.target_net = PMADQN_Net(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=5000)
        self.input_buffer = deque(maxlen=seq_len)
        
        self.gamma = 0.9
        self.alpha = 0.6  
        self.beta = 0.2   
        
        self.rho = 0.8    
        self.policy_stats = {
            'R': {0: 0.0, 1: 0.0}, 
            'E_counts': {0: 1e-5, 1: 1e-5}
        }
        self.prev_policy_idx = 0 

        self.batch_size = 128
        self.update_freq = 50
        self.step_counter = 0

    def save_checkpoint(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filename = os.path.join(directory, f"agent_{self.id}.pth")
        
        checkpoint = {
            'id': self.id,
            'q_net_state': self.q_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'policy_stats': self.policy_stats,
            'step_counter': self.step_counter
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, directory):
        filename = os.path.join(directory, f"agent_{self.id}.pth")
        if not os.path.exists(filename):
            print(f"Agent {self.id}: No checkpoint found at {filename}, starting from scratch.")
            return False

        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
            self.q_net.load_state_dict(checkpoint['q_net_state'])
            self.target_net.load_state_dict(checkpoint['target_net_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.policy_stats = checkpoint['policy_stats']
            self.step_counter = checkpoint['step_counter']
            return True
        except Exception as e:
            print(f"Agent {self.id}: Failed to load checkpoint. Error: {e}")
            return False

    def reset_state_buffer(self, initial_state):
        self.input_buffer.clear()
        zero_action = np.zeros(self.action_dim)
        init_input = np.concatenate([initial_state, zero_action])
        for _ in range(self.seq_len):
            self.input_buffer.append(init_input)

    def select_action(self, current_state, prev_action_idx, available_channels_mask=None):
        prev_action_vec = np.zeros(self.action_dim)
        if prev_action_idx is not None:
            prev_action_vec[prev_action_idx] = 1.0
            
        current_input = np.concatenate([current_state, prev_action_vec])
        self.input_buffer.append(current_input)
        
        input_seq = np.array(self.input_buffer) 
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, policy_probs, _ = self.q_net(input_tensor)
            
        q_values = q_values.cpu().data.numpy()[0]
        policy_probs = policy_probs.cpu().data.numpy()[0]
        
        if available_channels_mask is None:
            available_channels_mask = np.ones(self.action_dim)
        
        masked_q_values = q_values.copy()
        masked_q_values[available_channels_mask == 0] = -1e9
        
        valid_indices = np.where(available_channels_mask == 1)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(self.action_dim) 

        high_conf_indices = np.where(policy_probs > self.alpha)[0]
        cor_candidates = np.intersect1d(valid_indices, high_conf_indices)
        
        if len(cor_candidates) > 0:
            q_subset = masked_q_values[cor_candidates]
            action_cor = cor_candidates[np.argmax(q_subset)]
        else:
            action_cor = valid_indices[np.argmax(masked_q_values[valid_indices])]

        low_conf_indices = np.where(policy_probs < self.beta)[0]
        exp_candidates = np.intersect1d(valid_indices, low_conf_indices)
        
        if len(exp_candidates) > 0:
            action_exp = np.random.choice(exp_candidates)
        else:
            action_exp = np.random.choice(valid_indices)

        denom_0 = self.policy_stats['E_counts'][0]
        denom_1 = self.policy_stats['E_counts'][1]
        
        R0 = self.policy_stats['R'][0] / (denom_0 + 1e-5)
        R1 = self.policy_stats['R'][1] / (denom_1 + 1e-5)
        
        total_counts = denom_0 + denom_1 + 1e-5
        safe_log_val = np.log(max(total_counts, 1.0)) 
        bonus_0 = 1.0 * np.sqrt(safe_log_val / (denom_0 + 1e-5))
        bonus_1 = 1.0 * np.sqrt(safe_log_val / (denom_1 + 1e-5))
        
        Score_0 = R0 + bonus_0
        Score_1 = R1 + bonus_1 + 0.5 
        
        if Score_0 >= Score_1:
            final_action = action_cor
            selected_policy = 0
        else:
            final_action = action_exp
            selected_policy = 1
            
        self.prev_policy_idx = selected_policy
        return final_action

    def store_transition(self, state, action, reward, next_state, done):
        p_idx = self.prev_policy_idx
        
        self.policy_stats['R'][p_idx] = self.rho * self.policy_stats['R'][p_idx] + (1 - self.rho) * reward
        self.policy_stats['E_counts'][p_idx] += 1.0 
        
        current_seq = np.array(self.input_buffer)
        self.memory.append((current_seq, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_seq_batch, actions, rewards, next_states, dones = zip(*batch)

        state_seq = torch.FloatTensor(np.array(state_seq_batch)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        next_seq_list = []
        for i in range(len(state_seq_batch)):
            curr = state_seq_batch[i]
            nxt_s = next_states[i]
            new_seq = np.roll(curr, -1, axis=0)
            zero_act = np.zeros(self.action_dim)
            new_seq[-1] = np.concatenate([nxt_s, zero_act])
            next_seq_list.append(new_seq)
            
        next_state_seq = torch.FloatTensor(np.array(next_seq_list)).to(self.device)

        q_values_full, policy_probs, _ = self.q_net(state_seq)
        
        q_eval = q_values_full.gather(1, actions)

        with torch.no_grad():
            q_next, _, _ = self.target_net(next_state_seq)
            q_next = q_next.max(1, keepdim=True)[0] 
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss_q = nn.MSELoss()(q_eval, q_target)

        with torch.no_grad():
            best_actions = q_values_full.argmax(dim=1) 
        
        loss_p = nn.NLLLoss()(torch.log(policy_probs + 1e-9), best_actions)
        
        loss = loss_q + 0.5 * loss_p

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.step_counter += 1
        if self.step_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
