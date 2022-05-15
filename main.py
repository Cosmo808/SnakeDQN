from utils import plot, save_np
import matplotlib.pyplot as plt
import time
from game import SnakeGameAI
from agent import Agent
import numpy as np


deep_flag = True
load_flag = False
plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0
agent = Agent(load_flag)
game = SnakeGameAI()
plt.ion()
t = time.time()

while True:
    # get old state
    state_old = agent.get_state(game)

    # get move
    if deep_flag:
        final_move = agent.dqn_get_action(state_old)
    else:
        final_move = agent.ql_get_action(state_old)

    # perform move and get new state
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state(game)

    # train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done, deep_flag)

    # remember
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        # train long memory, plot the result
        game.reset()
        agent.n_games += 1
        agent.train_long_memory(deep_flag)

        if score > record:
            record = score
            if deep_flag:
                agent.dqn_model.save(file_name=str(record)+'.pth')
            else:
                agent.Q_table_save(file_name=str(record)+'.npy')

        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        if agent.n_games % 100 == 0:
            print('\nEpoch {:d}: {:.0f} seconds\n'.format(agent.n_games, time.time() - t))

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)

        if agent.n_games % 300 == 0:
            save_np('./Scores', 'score.npy', np.array(plot_scores))
            save_np('./Scores', 'mean.npy', np.array(plot_mean_scores))
            print("\nSaving results...\n")
