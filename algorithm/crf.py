import numpy as np
import pickle
from scipy.optimize import fmin_l_bfgs_b

START_TAG = "<START>"
END_TAG = "<END>"
START_INDEX = None
END_INDEX = None


class FeatureManager:
    def __init__(self, state_num: int, feature_funcs: list, feature_weights: list):
        self.state_num = state_num
        self.feature_funcs = feature_funcs
        self.feature_weights = np.array(feature_weights)
        self.feature_num = len(feature_funcs)

    def _compute_w(self, y_prev, y, x, index):
        h = np.zeros(len(self.feature_funcs), dtype=np.float)
        for i, f in enumerate(self.feature_funcs):
            if f(y_prev, y, x, index):
                h[i] = 1.0
        return np.dot(h, self.feature_weights), h

    def compute_m(self, x, inference=True):
        m = []
        h = dict()
        for i in range(len(x) + 1):
            table = np.zeros(shape=(self.state_num, self.state_num), dtype=np.float)
            for y_prev in range(self.state_num):
                for y in range(self.state_num):
                    table[y_prev, y], hit = self._compute_w(y_prev, y, x, i)
                    if not inference:
                        h[(y_prev, y, i)] = hit
            table = np.exp(table)
            if i == 0:
                temp = table[START_INDEX, :]
                table[:, :] = 0.0
                table[START_INDEX, :] = temp
            elif i == len(x):
                temp = table[:, END_INDEX]
                table[:, :] = 0.0
                table[:, END_INDEX] = temp
            else:
                table[START_INDEX, :] = 0.0
                table[:, END_INDEX] = 0.0
            m.append(table)
        if inference:
            return m
        else:
            return m, h

    def compute_transform(self, x, index):
        table = np.ones(shape=(self.state_num, self.state_num), dtype=np.float)
        for y_prev in range(self.state_num):
            for y in range(self.state_num):
                table[y_prev, y] = self._compute_w(y_prev, y, x, index)
        if index == 0:
            temp = table[START_INDEX, :]
            table[:, :] = 0.0
            table[START_INDEX, :] = temp
        return table

    def update_feature_weights(self, weights):
        assert len(weights) == self.feature_num
        self.feature_weights = weights


class CRF:
    def __init__(self, states: list = None, feature_manager: FeatureManager = None):
        self.states = None
        self.state_num = None
        self.state_vocab = None
        self.feature_manager = feature_manager
        if states is not None:
            self.init_states(states)

    def init_states(self, states: list):
        self.states = states
        if START_TAG not in self.states:
            self.states = [START_TAG] + self.states
        if END_TAG not in self.states:
            self.states += [END_TAG]
        self.state_num = len(self.states)
        self.state_vocab = {k: v for k, v in enumerate(self.states)}

        global START_INDEX, END_INDEX
        START_INDEX = self.state_vocab[START_TAG]
        END_INDEX = self.state_vocab[END_TAG]

    def reset_param(self, feature_manager: FeatureManager, save_file: str = None):
        if save_file is None:
            self.feature_manager = feature_manager
        else:
            with open(save_file) as f:
                data = pickle.load(f)
            self.feature_manager = data["feature_manager"]
            self.init_states(data["states"])

    def _check_param(self):
        if self.feature_manager is not None and self.states is not None:
            pass
        else:
            raise Exception("参数未初始化！")

    def _forward(self, m: list):
        # m = [M(0, l * l), M(1, l * l), ...]
        alpha = np.ones(shape=(len(m), self.state_num), dtype=np.float)
        for i in range(len(m)):
            if i == 0:
                continue
            alpha[i] = np.matmul(alpha[i - 1, :], m[i])
        return alpha

    def _backward(self, m: list):
        beta = np.ones(shape=(len(m), self.state_num), dtype=np.float)
        for i in range(len(m) - 1, -1, -1):
            if i == len(m) - 1:
                continue
            beta[i] = np.matmul(m[i], beta[i + 1])
        return beta

    def _vitibi(self, x):
        _states = np.zeros(shape=(len(x), self.state_num), dtype=np.int)
        _values = np.zeros(shape=(len(x), self.state_num), dtype=np.float)
        for i in range(len(x)):
            t = self.feature_manager.compute_transform(x, i)
            if i == 0:
                # _states[i] = np.max(t, axis=-1)
                _values[i] = np.argmax(t, axis=-1)
                continue
            t = np.expand_dims(_values[i - 1], axis=-1) + t
            _states[i] = np.max(t, axis=-1)
            _values[i] = np.argmax(t, axis=-1)

        index = np.argmax(_values[len(x) - 1])
        value = _values[len(x) - 1, index]
        states = list(_states[1:])
        states += [index]

        return states, np.float(value)

    def log_likelihood(self, xs, ys, weights, l2_coef=0.0):
        func_expected_value = np.zeros(self.feature_manager.feature_num, dtype=np.float)
        func_empirical_value = np.zeros(self.feature_manager.feature_num, dtype=np.float)
        total_log_z = 0.0
        self.feature_manager.update_feature_weights(weights)
        for x, y in zip(xs, ys):
            m, h = self.feature_manager.compute_m(x, inference=False)
            alpha = self._forward(m)
            beta = self._backward(m)
            z = np.sum(np.exp(alpha))
            total_log_z += np.log(z)
            for t in range(len(x) + 1):
                if t == 0 or t == len(x):
                    continue
                else:
                    prob = np.exp(alpha[t - 1, y[t - 1]] * m[t - 1][t - 1, t] * beta[t, y[t]]) / z
                    if (y[t - 1], y[t], t) in h:
                        func_expected_value += h[(y[t - 1], y[t], t)] * prob
                        func_empirical_value += h[(y[t - 1], y[t], t)]
        likelihood = total_log_z - np.dot(func_empirical_value, weights) + l2_coef * np.dot(weights, weights)
        gs = func_expected_value - func_empirical_value + l2_coef * weights

        return likelihood, gs

    def predict(self, x: list) -> (list, float):
        self._check_param()
        if len(x) == 0:
            return [], 0.0

        states, value = self._vitibi(x)

        return [self.states[i] for i in states], value

    @staticmethod
    def learn(w, crf, xs, ys):
        return crf.log_likelihood(xs, ys, w)


def train(crf: CRF, xs, ys):
    weights = np.zeros(crf.feature_manager.feature_num, dtype=np.float)
    weights, value, info = fmin_l_bfgs_b(CRF.learn, weights, args=(crf, xs, ys))
    crf.feature_manager.update_feature_weights(weights)
