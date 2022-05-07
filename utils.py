import matplotlib.pyplot as plt
from IPython import display


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.cla()
    plt.title('Training')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.1)


def s2idx(state):
    index = 0
    for i, j in enumerate(state):
        if j:
            ii = 1
        else:
            ii = 0
        index += 2 ** i * ii
    return index
