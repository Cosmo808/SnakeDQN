import matplotlib.pyplot as plt
from IPython import display
import os
import numpy as np


display.clear_output(wait=True)
display.display(plt.gcf())
plt.cla()


# smooth func
def smooth_curve(points, factor=0.90):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def ql_dqn_plot():
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

    # data smooth
    dqn_score = smooth_curve(dqn_score)
    ql_score = smooth_curve(ql_score)

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


def var_plot(director):
    var_mean = {}
    var_score = {}
    index_sheet = []
    for root, dirs, files in os.walk(director):
        if files:
            for file in files:
                index = root[len(director)+1:]
                if not index_sheet or index != index_sheet[-1]:
                    index_sheet.append(index)
                if file == 'mean.npy':
                    mean = np.load(os.path.join(root, file))
                    var_mean[index] = mean
                elif file == 'score.npy':
                    score = np.load(os.path.join(root, file))
                    score = smooth_curve(score)
                    var_score[index] = score

    plt.figure(1)
    plt.title('DQN Variable')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Value')
    for i in index_sheet:
        mean_value = var_mean[i]
        plt.plot(mean_value)
    plt.legend(index_sheet)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1500)

    plt.figure(2)
    plt.title('DQN Variable')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    for i in index_sheet:
        score_value = var_score[i]
        plt.plot(score_value)
    plt.legend(index_sheet)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1500)
    plt.show()


if __name__ == '__main__':
    var_plot('./algorithms')
