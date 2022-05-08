import numpy as np


def to_transform(vocab, targets, unknown_value):
    sources = []
    for t in targets:
        if t in vocab:
            s = vocab[t]
        else:
            s = unknown_value
        sources.append(s)
    return sources


def random_vector(size):
    r = np.random.uniform(0, 1, size=size)
    s = np.sum(r)
    return r / s


def laplace_smoothing(x, y, c, smooth=1e-6):
    return (x + smooth) / (y + c * smooth)


class HMM:
    def __init__(self, state_vocab: dict, observation_vocab: dict, debug=False):
        # A:状态转移矩阵
        # B:观察矩阵
        # pi:状态初始化向量
        self.state_vocab = state_vocab
        self.inverse_state_vocab = {v: k for k, v in state_vocab.items()}
        self.observation_vocab = observation_vocab
        self.state_num = len(state_vocab)
        self.observation_num = len(observation_vocab)
        self.A = None
        self.B = None
        self.pi = None
        self.debug = debug

    def reset_param(self, A=None, B=None, pi=None, save_file: str = None):
        if A is not None and B is not None and pi is not None:
            self.A = A
            self.B = B
            self.pi = pi
        elif save_file:
            pass

    def param_initialization(self):
        self.pi = random_vector(self.state_num)
        _A, _B = [], []
        for i in range(self.state_num):
            _A.append(np.expand_dims(random_vector(self.state_num), axis=0))
            _B.append(np.expand_dims(random_vector(self.observation_num), axis=0))
        self.A = np.concatenate(_A, axis=0)
        self.B = np.concatenate(_B, axis=0)

    def _check_param(self):
        if self.A is not None and self.B is not None and self.pi is not None:
            pass
        else:
            raise Exception("参数未初始化！")

    # 前向算法
    def _forword(self, o: list):
        alpha = np.zeros(shape=(len(o), self.state_num), dtype=np.float)
        for i, _o in enumerate(o):
            if i == 0:
                alpha[i] = np.multiply(self.pi, self.B[:, _o])
            else:
                b = np.expand_dims(self.B[:, _o], 0)
                _alpha = np.multiply(np.matmul(alpha[i - 1], self.A), b)
                alpha[i] = _alpha[0]
        if self.debug:
            print("alpha:\n{}".format(alpha))
        return alpha

    # 后向算法
    def _backword(self, o: list):
        beta = np.ones(shape=(len(o), self.state_num), dtype=np.float)
        for i in range(len(o) - 2, -1, -1):
            _beta = np.expand_dims(np.multiply(self.B[:, o[i + 1]], beta[i + 1]), -1)
            _beta = np.matmul(self.A, _beta)
            beta[i] = np.transpose(_beta)[0]
        if self.debug:
            print("beta:\n{}".format(beta))

        return beta

    # 维特比算法
    def _vitibi(self, o: list):
        _states = np.zeros(shape=(len(o), self.state_num), dtype=np.int)
        _probs = np.zeros(shape=(len(o), self.state_num), dtype=np.float)

        for i, _o in enumerate(o):
            if i == 0:
                _probs[i] = np.multiply(self.pi, self.B[:, _o])
                continue
            _prob = np.multiply(_probs[i - 1], self.A)
            _states[i] = np.argmax(_prob, axis=-1)
            _probs[i] = np.multiply(np.max(_prob, axis=-1), self.B[:, _o])

        if self.debug:
            print("probs:\n{}".format(_probs))
            print("states:\n{}".format(_states))

        index = np.argmax(_probs[len(o) - 1])
        prob = _probs[len(o) - 1, index]
        state = _states[:, index]
        state = list(state[1:]) + [index]

        return state, np.float(prob)

    # 概率问题
    def observation_probability(self, o: list, mode: str = "F") -> float:
        self._check_param()
        if len(o) == 0:
            return 0.0

        o = to_transform(self.observation_vocab, o, 0)
        if self.debug:
            print("O:{}".format(o))
        if mode in ["F", "f"]:
            d = self._forword(o)[-1]
        elif mode in ["B", "b"]:
            d = self._backword(o)[0]
            d = np.multiply(self.pi, np.multiply(self.B[:, o[0]], d))
        else:
            raise Exception("未知\"{}\"！".format(mode))
        return np.float(np.sum(d))

    # 学习问题
    def learn(self, o_list: list, smooth: float = 1e-6):
        _pi = np.zeros(self.state_num, dtype=np.float)
        _A = np.zeros((self.state_num, self.state_num), dtype=np.float)
        _B = np.zeros((self.state_num, self.observation_num), dtype=np.float)
        for o in o_list:
            seq_length = len(o)
            # E step
            alpha = self._forword(o)
            beta = self._backword(o)
            r = np.multiply(alpha, beta)
            # M step
            for i in range(self.state_num):
                # 计算pi
                # _pi[i] = r[0, i] / np.sum(r[0])
                _pi[i] += laplace_smoothing(r[0, i], np.sum(r[0]), self.state_num, smooth)
                # 计算A
                p_i = 0.0
                for t in range(seq_length - 1):
                    p_i += r[t, i]
                for j in range(self.state_num):
                    p_i_j = 0.0
                    for t in range(seq_length - 1):
                        p_i_j += alpha[t, i] * self.A[i, j] * self.B[j, o[t + 1]] * beta[t, j]
                    # _A[i, j] = p_i_j / p_i
                    _A[i, j] += laplace_smoothing(p_i_j, p_i, self.state_num, smooth)
                # 计算B
                p_i_t = p_i + r[seq_length - 1, i]
                _b = [0.0 for i in range(self.observation_num)]
                for t in range(seq_length):
                    _b[o[t]] += r[t, i]
                for j in range(self.observation_num):
                    # _B[i, j] = _b[j] / p_i_t
                    _B[i, j] += laplace_smoothing(_b[j], p_i_t, self.observation_num, smooth)
        # 更新参数
        _pi /= len(o_list)
        _A /= len(o_list)
        _B /= len(o_list)
        self.reset_param(A=_A, B=_B, pi=_pi)

    # 预测问题
    def predict(self, o: list) -> (list, float):
        self._check_param()
        if len(o) == 0:
            return [], 0.0

        o = to_transform(self.observation_vocab, o, 0)
        s, prob = self._vitibi(o)

        return to_transform(self.inverse_state_vocab, s, 1), prob


def train(data: dict, epoch=10):
    pass


def test():
    state_vocab = {1: 0, 2: 1, 3: 2}
    observation_vocab = {"红": 0, "白": 1}
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    pi = [0.2, 0.4, 0.4]
    A, B, pi = np.array(A), np.array(B), np.array(pi)

    hmm = HMM(state_vocab, observation_vocab, debug=True)
    # print(hmm.param_initialization())
    hmm.reset_param(A=A, B=B, pi=pi)
    O = ["红", "白", "红"]
    prob = hmm.observation_probability(O, mode="f")
    print(prob)

    indices = hmm.predict(O)
    print(indices)


if __name__ == "__main__":
    test()
