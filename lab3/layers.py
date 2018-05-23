import numpy as np

class Layer():
    def __init__(self, in_size, out_size, lam, name):
        self.in_size = in_size
        self.out_size = out_size
        self.name = name
        self.isActivation = (False if name is "linear" else True)

        if not self.isActivation:
            # Weights
            self.W = np.random.normal(loc=0.0, scale=0.01, size=(out_size, in_size))
            # Bias
            self.b = np.random.normal(loc=0.0, scale=0.01, size=(out_size, 1))
            # Weight regularization
            self.lam = lam
            self.mom = {
                'W' : np.zeros(self.W.shape),
                'b' : np.zeros(self.b.shape)
            }

            self.resetGrad()
        # this is a memory variable between forward/backward
        self.x = np.empty(shape=(self.in_size, 1))


    def forward(self, x):
        assert x is not None
        # sometimes we need to store input for backward
        self.x = x
        #print(self.name + " forward")
        #print(self.x.shape)

    def backward(self):
        assert self.x is not None
        #print(self.name + " back")
        #print(self.x.shape)

    def cost(self):
        return 0

    def resetGrad(self):
        self.gW = np.zeros(self.W.shape)
        self.gB = np.zeros(self.b.shape)

    # for non-activation layers to implement
    def update(self, l_rate=0.001):
        pass

    def updateMom(self, l_rate=0.001, momentum=0.0):
        pass

class Linear(Layer):
    def __init__(self, in_size, out_size, lam=0, name="linear"):
        super().__init__(in_size, out_size, lam, name)

    def forward(self, x):
        Layer.forward(self, x)
        # Wx + b
        return np.dot(self.W, x) + self.b

    def backward(self, grad):
        Layer.backward(self)
        N = self.x.shape[1]

        self.resetGrad()

        for i in range(N):
            p = self.x[:, i]
            g = grad[i, :]

            self.gW += np.outer(g, p)
            self.gB += np.reshape(g, self.gB.shape)

        # here's the difference in (10) and (11)
        self.gW = (1.0/N) * self.gW + 2 * self.lam * self.W
        self.gB /= N

        return np.dot(grad, self.W)

    def cost(self):
        return self.lam * np.power(self.W, 2).sum()

    def update(self, l_rate=0.001):
        self.W -= l_rate * self.gW
        self.b -= l_rate * self.gB

    def updateMom(self, l_rate=0.001, momentum=0.0):
        self.mom['W'] = momentum * self.mom['W'] + l_rate * self.gW
        self.mom['b'] = momentum * self.mom['b'] + l_rate * self.gB

        self.W -= self.mom['W']
        self.b -= self.mom['b']

class ReLU(Layer):
    def __init__(self, in_size, name="relu"):
        super().__init__(in_size, in_size, -1, name)

    def forward(self, x):
        Layer.forward(self, x)
        # max(0, x)
        return self.x * (self.x > 0)

    def backward(self, grad):
        Layer.backward(self)
        return np.multiply(grad, self.x.T > 0)

class Softmax(Layer):
    def __init__(self, in_size, name="softmax"):
        super().__init__(in_size, in_size, -1, name)

    def forward(self, x):
        assert x is not None
        try:
            # this should prevent error tried for
            e = np.exp(x - x.max())
            res = e / np.sum(e, axis=0)
        except FloatingPointError:
            # Gradient explosion scenario
            print("jesus take the wheel")
            res = np.ones(x)
        Layer.forward(self, res)
        return res

    def backward(self, truth):
        Layer.backward(self)
        assert self.x.shape[1] == truth.shape[1]
        N = truth.shape[1]

        cols = ((truth[:,i], self.x[:,i]) for i in range(N))
        grad = [self.softGrad(t, p) for (t, p) in cols]

        return np.vstack(grad)

    @staticmethod
    def softGrad(t, p):
        # Jacobian according for formulas in Ass1
        a = np.outer(p,p)
        b = np.dot(t, (np.diag(p) - a))
        c = np.dot(t, p)
        return -b/c

    def cost(self, truth, prob=None):
        x = self.x if prob is None else prob
        assert x.shape[1] == truth.shape[1]
        N = x.shape[1]

        Py = np.multiply(truth, x).sum(axis=0)
        Py[Py == 0] = np.finfo(float).eps # fix floats

        return - np.log(Py).sum() / N

class BatchNorm(Layer):
    # https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    def __init__(self, in_size, mu=None, s=None, alpha=0.99, name="batch_norm"):
        super().__init__(in_size, in_size, -1, name)

        self.x = np.empty(shape=(self.in_size, 1))
        self.mu = np.zeros(shape=(in_size, 1), dtype=float)
        self.s = np.ones(shape=(in_size, 1), dtype=float)

        self.avg_mu = mu
        self.avg_s = s
        self.alpha = alpha
        # fix floats
        if self.avg_mu is not None:
            self.avg_mu[self.avg_mu == 0] = np.finfo(float).eps
        if self.avg_s is not None:
            self.avg_s[self.avg_s == 0] = np.finfo(float).eps

    def forward(self, x, train=False):
        Layer.forward(self, x)
        self.x = x if train else self.x
        self.mu = x.mean(axis=1, keepdims=True) if train else self.avg_mu
        self.s = x.var(axis=1, keepdims=True) if train else self.avg_s
        # fix floats
        self.mu[self.mu == 0] = np.finfo(float).eps
        self.s[self.s == 0] = np.finfo(float).eps
        # running avg
        self.avg_mu = self.mu if self.avg_mu is None else self.avg_mu
        self.avg_s = self.s if self.avg_s is None else self.avg_s

        self.mu = (self.alpha * self.avg_mu) + (1.0 - self.alpha) * self.mu
        self.s = (self.alpha * self.avg_s) + (1.0 - self.alpha) * self.s

        return self.s ** -0.5 * (x - self.mu)

    def backward(self, grad):
        Layer.backward(self)
        N = self.x.shape[1]

        # Begin formula land, lecture 4
        gT = grad.T
        inv_sqrt_mu = self.s ** -0.5
        #print(inv_sqrt_mu)
        x_center = self.x - self.mu

        dJ_dv = (self.s ** -1.5) * (gT * x_center).sum(axis=1, keepdims=True)
        dJ_dMu = - inv_sqrt_mu * gT.sum(axis=1, keepdims=True)

        gT = gT * inv_sqrt_mu + dJ_dv * (x_center / N) + (dJ_dMu / N)

        return gT.T # back to normal
