import matplotlib.pyplot as plt
from IPython import display
import os
import numpy as np


display.clear_output(wait=True)
display.display(plt.gcf())
plt.cla()


# smooth func
def smooth_curve(points, factor=0.93):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# load data
dqn_mean = []
dqn_score = []
ql_mean = []
ql_score = []
for root, dirs, files in os.walk('./'):
    for file in files:
        if 'dqn' in root:
            if file == 'mean.npy':
                dqn_mean = np.load(os.path.join(root, file))
            elif file == 'score.npy':
                dqn_score = np.load(os.path.join(root, file))
        elif 'ql' in root:
            if file == 'mean.npy':
                ql_mean = np.load(os.path.join(root, file))
            elif file == 'score.npy':
                ql_score = np.load(os.path.join(root, file))
# data preprocess
ql_mean = ql_mean[:1500]
ql_score = ql_score[:1500]
dqn_score = smooth_curve(dqn_score)
ql_score = smooth_curve(ql_score)


dqn_var_mean = {}
dqn_var_score = {}
for root, dirs, files in os.walk('./dqn'):
    if files:
        for file in files:
            dirt = root
            alpha = dirt[12:15]
            gamma = dirt[-3:]
            if file == 'mean.npy':
                mean = np.load(os.path.join(root, file))
                mean = smooth_curve(mean)
                dqn_var_mean['alpha={:s} gamma={:s}'.format(alpha, gamma)] = mean
            elif file == 'score.npy':
                score = np.load(os.path.join(root, file))
                score = smooth_curve(score)
                dqn_var_score['alpha={:s} gamma={:s}'.format(alpha, gamma)] = score


def ql_dqn_plot():
    # plot mean value
    plt.figure(1)
    plt.title('DQN & QLearning')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Value')
    plt.plot(dqn_mean)
    plt.plot(ql_mean)
    plt.legend(['DQN', 'QLearning'])
    plt.text(1000, 1, 'alpha=0.2, gamma=0.9')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1500)

    plt.figure(2)
    plt.title('DQN & QLearning')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.plot(dqn_score)
    plt.plot(ql_score)
    plt.legend(['DQN', 'QLearning'], loc='upper left')
    plt.text(1000, 1, 'alpha=0.2, gamma=0.9')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1500)
    plt.show()


def dqn_var_plot():
    alpha_sheet = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0]
    gamma_sheet = [0.9, 0.9, 0.9, 0.9, 0.7, 0.9]

    plt.figure(1)
    plt.title('DQN Variable')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Value')
    index = []
    for a, g in zip(alpha_sheet, gamma_sheet):
        index.append('alpha={:s} gamma={:s}'.format(str(a), str(g)))
        mean_value = dqn_var_mean[index[-1]]
        plt.plot(mean_value)
    plt.legend(index)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1500)

    plt.figure(2)
    plt.title('DQN Variable')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    index = []
    for a, g in zip(alpha_sheet, gamma_sheet):
        index.append('alpha={:s} gamma={:s}'.format(str(a), str(g)))
        score_value = dqn_var_score[index[-1]]
        plt.plot(score_value)
    plt.legend(index)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1500)
    plt.show()


if __name__ == '__main__':
    dqn_var_plot()
