import torch
import torch.nn as nn
import torch.nn.functional as F

class PMADQN_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=1):
        super(PMADQN_Net, self).__init__()
        
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=hidden_dim, 
                            num_layers=lstm_layers, batch_first=True)
        
        self.fc_q1 = nn.Linear(hidden_dim, 64)
        self.fc_q2 = nn.Linear(64, action_dim)
        
        self.fc_p1 = nn.Linear(hidden_dim, 64)
        self.fc_p2 = nn.Linear(64, action_dim)

    def forward(self, x, hidden=None):
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        final_feature = lstm_out[:, -1, :]
        
        q = F.relu(self.fc_q1(final_feature))
        q_values = self.fc_q2(q)
        
        p = F.relu(self.fc_p1(final_feature))
        policy_probs = F.softmax(self.fc_p2(p), dim=-1)
        
        return q_values, policy_probs, new_hidden
