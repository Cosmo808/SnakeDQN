import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        model_folder_path = './Model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class target_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Nature_Trainer:
    def __init__(self, qnet, target_qnet, lr, alpha, gamma, device):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.qnet = qnet
        self.target_qnet = target_qnet
        self.optimizer = optim.Adam(qnet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device
        self.target_set_iterations = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        predict = self.qnet(state)
        target = predict.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_qnet(next_state[idx]))
            Q_old = target[idx][torch.argmax(action[idx]).item()]
            target[idx][torch.argmax(action[idx]).item()] = Q_old + self.alpha * (Q_new - Q_old)

        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()
        self.optimizer.step()

        self.target_set_iterations += 1
        if self.target_set_iterations >= 0:
            self.target_set_iterations = 0
            self.target_qnet.load_state_dict(self.qnet.state_dict())
