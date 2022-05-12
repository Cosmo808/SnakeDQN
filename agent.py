import torch
import random
from collections import deque
from game import Direction, Point
from utils import s2idx
import os
import numpy as np
from ql_model import QLTrainer
from dqn_model import Linear_QNet, QTrainer
from nature_dqn_model import Qnet, target_Qnet, NTrainer


MAX_MEMORY = 10000
BATCH_SIZE = 1024


class Agent:

    def __init__(self, load):
        # hyper parameters
        self.n_games = 0
        self.epsilon = 0.1
        self.alpha = 1.0
        self.gamma = 0.7
        self.lr = 0.001
        self.memory = deque(maxlen=MAX_MEMORY)

        # cuda or cpu
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('\nGPU Accelerating...\n')
        else:
            print('\nUsing CPU...\n')

        # q-learning model
        self.ql = QLTrainer(self.alpha, self.gamma)
        self.Q = np.zeros((2 ** 11, 3))

        # dqn model
        self.dqn_model = Linear_QNet(11, 256, 3, self.device).to(self.device)
        self.dqn_trainer = QTrainer(self.dqn_model, self.lr, self.alpha, self.gamma, self.device)

        # nature dqn
        self.qnet = Qnet(11, 256, 3, self.device).to(self.device)
        self.target_qnt = target_Qnet(11, 256, 3, self.device).to(self.device)
        self.n_trainer = NTrainer(self.qnet, self.target_qnt, self.lr, self.alpha, self.gamma, self.device)

        # load model
        if load:
            self._load_dnn()
            self._load_Q()

    def train_long_memory(self, deep_flag):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        if deep_flag:
            self.n_trainer.train_step(states, actions, rewards, next_states, dones)
        else:
            self.Q = self.ql.ql_train(states, actions, rewards, next_states, dones, self.Q)

    def train_short_memory(self, state, action, reward, next_state, done, deep_flag):
        if deep_flag:
            self.n_trainer.train_step(state, action, reward, next_state, done)
        else:
            self.Q = self.ql.ql_train(state, action, reward, next_state, done, self.Q)

    def get_action(self, state):
        final_move = [0, 0, 0]
        if self.n_games > 80:
            self.epsilon = 0
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.dqn_model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def ql_get_action(self, state):
        final_move = [0, 0, 0]
        if self.n_games < 40:
            self.epsilon = 0.005
        else:
            self.epsilon = 0.001
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            index = s2idx(state)
            move = self.Q[index]
            action = np.argmax(move)
            final_move[int(action)] = 1
        return final_move

    def Q_table_save(self, file_name):
        Q_table_path = './Q_table'
        if not os.path.exists(Q_table_path):
            os.makedirs(Q_table_path)
        file_name = os.path.join(Q_table_path, file_name)
        np.save(file_name, self.Q)

    def _load_dnn(self):
        model_dir = './Model'
        record = 0
        for root, dirs, files in os.walk(model_dir):
            if not files:
                return
            for file in files:
                score = int(file.split('.')[0])
                if score > record:
                    record = score
        model_name = './Model/' + str(record) + '.pth'
        self.dqn_model.load_state_dict(torch.load(model_name))
        print('Loading Model: {:s} ...\n'.format(str(record) + '.pth'))

    def _load_Q(self):
        Q_table_dir = './Q_table'
        record = 0
        for root, dirs, files in os.walk(Q_table_dir):
            if not files:
                return
            for file in files:
                score = int(file.split('.')[0])
                if score > record:
                    record = score
        Q_table_name = './Q_table/' + str(record) + '.npy'
        self.Q = np.load(Q_table_name)
        print('Loading Q Table: {:s} ...\n'.format(str(record) + '.npy'))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def get_state(game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y < game.head.y,  # food down
        ]

        return state
