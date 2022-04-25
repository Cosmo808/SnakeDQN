import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from torch_model import Linear_QNet, QTrainer
from helper import plot
import matplotlib.pyplot as plt
from keras_model import DNN
import tensorflow as tf


MAX_MEMORY = 10000
BATCH_SIZE = 512
LR = 0.01


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        # self.memory = deque(maxlen = MAX_MEMORY)  #popleft()

        self.model = Linear_QNet(11, 256, 3)  # input_size: 11, output: 3
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

        self.dnn = DNN(state_size=11, action_size=3, lr=1e-2, gamma=0.9)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MAX_MEMORY, self.dnn.state_size * 2 + 3))
        self.replace_target_iter = 50

    def get_state(self, game):
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

            # Move dirction
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

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        action = np.argmax(action)
        # self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached
        transition = np.hstack((state, action, reward, next_state, done))
        index = self.memory_counter % MAX_MEMORY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train_long_memory(self):
        # if len(self.memory) > BATCH_SIZE:
        #     mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        # else:
        #     mini_sample = self.memory

        # states, actions, rewards, next_states, dones = zip(*mini_sample)
        # self.trainer.train_step(states, actions, rewards, next_states, dones)

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.dnn.target_replace_op()
            print('\ntarget_params_replaced\n')

        if self.memory_counter > MAX_MEMORY:
            sample_index = np.random.choice(MAX_MEMORY, size = BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, size = BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        q_next = self.dnn.model_targ.predict(
            batch_memory[:, -self.dnn.state_size - 1:-1])  # use next state to predict next Q
        q_eval = self.dnn.model_eval.predict(batch_memory[:, :self.dnn.state_size])  # use state to predict evaluation Q
        q_targ = q_eval.copy()

        batch_index = np.arange(BATCH_SIZE, dtype = np.int32)
        eval_action_index = batch_memory[:, self.dnn.state_size].astype(int)
        reward = batch_memory[:, self.dnn.state_size + 1]
        q_targ[batch_index, eval_action_index] = reward + self.gamma * np.max(q_next, axis = 1)

        self.dnn.model_eval.fit(
            x = batch_memory[:, :self.dnn.state_size], y = q_targ, epochs = 10, verbose=0
        )
        self.learn_step_counter += 1

    def train_short_memory(self, state, action, reward, next_state, done):
        # self.trainer.train_step(state, action, reward, next_state, done)
        action = np.argmax(action)
        state = tf.expand_dims(state, axis = 0)
        next_state = tf.expand_dims(next_state, axis = 0)
        q_next = self.dnn.model_targ.predict(next_state)
        q_eval = self.dnn.model_targ.predict(state)
        q_targ = q_eval.copy()
        q_targ[0, action] = reward + self.gamma * np.max(q_next, axis = 1)

        self.dnn.model_eval.fit(
            x = state, y = q_targ, epochs = 3, verbose=0
        )
        self.learn_step_counter += 1


    def get_action(self, state):
        # # random moves : tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games
        # final_move = [0, 0, 0]
        # if random.randint(0, 200) < self.epsilon:
        #     move = random.randint(0, 2)
        #     final_move[move] = 1
        # else:
        #     state0 = torch.tensor(state, dtype = torch.float)
        #     prediction = self.model(state0)
        #     move = torch.argmax(prediction).item()
        #     final_move[move] = 1
        #
        # return final_move

        action = np.zeros(3)
        state = np.array(state)
        state = tf.expand_dims(state, axis = 0)
        self.epsilon = (200 - self.n_games) / 300
        if self.n_games > 170:
            self.epsilon = 0.1
        if random.random() > self.epsilon:
            action_value = self.dnn.model_targ.predict(state)
            ind = np.argmax(action_value)  # [0, 2]
            action[ind] = 1
        else:
            ind = random.randint(0, 2)
            action[ind] = 1
        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    plt.ion()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
