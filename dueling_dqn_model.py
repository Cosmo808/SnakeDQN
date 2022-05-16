import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Dueling_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(Dueling_network, self).__init__()
        self.fc_value = nn.Linear(input_size, hidden_size)
        self.fc_adv = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        self.adv = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))
        value = self.value(value)
        adv = self.adv(adv)

        x = value + adv - adv.mean()
        return x


class Dueling_Trainer:
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
