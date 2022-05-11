import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from utils import s2idx


class Linear_QNet(nn.Module):
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


class QTrainer:
    def __init__(self, model, lr, alpha, gamma, device):
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device

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

        predict = self.model(state)
        target = predict.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            Q_old = target[idx][torch.argmax(action[idx]).item()]
            target[idx][torch.argmax(action[idx]).item()] = Q_old + self.alpha * (Q_new - Q_old)

        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()
        self.optimizer.step()

    def ql_train(self, state, action, reward, next_state, done, Q):
        state = np.array(state)
        next_state = np.array(next_state)
        if len(state.shape) == 1:
            state = [state]
            action = [action]
            reward = [reward]
            next_state = [next_state]
            done = (done,)
        for idx in range(len(done)):
            index = s2idx(state[idx])
            index_next = s2idx(next_state[idx])
            r = reward[idx]
            a = action[idx]
            Q_new = r
            if not done[idx]:
                Q_new = r + self.gamma * np.max(Q[index_next])
            Q_old = Q[index][np.argmax(a)]
            Q[index][np.argmax(a)] = Q_old + self.alpha * (Q_new - Q_old)

        return Q
