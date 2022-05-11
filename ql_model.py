import numpy as np
from utils import s2idx


class QLTrainer:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

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
