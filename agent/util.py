import numpy as np
from collections import deque

class Memory:
    def __init__(self, memory_size, observation_space):
        self.head = 0
        self.is_full = False
        self.memory_size = memory_size
        #self.obs_buf  = np.empty((memory_size,) + observation_space) #empty's parameter is a tuple
        self.obs_buf = np.empty((memory_size,), dtype='object')
        self.act_buf  = np.empty((memory_size,), dtype=np.int16)
        # self.act_buf = deque(maxlen=self.memory_size)
        self.rwd_buf  = np.empty((memory_size,), dtype=np.float32)
        #self.nobs_buf = np.empty((memory_size,) + observation_space)
        self.nobs_buf = np.empty((memory_size,), dtype='object')

    def add(self, obs, act, rwd, nobs):
        self.obs_buf[self.head]  = obs
        self.act_buf[self.head]  = act
        # self.act_buf.append(act)
        self.rwd_buf[self.head]  = rwd
        self.nobs_buf[self.head] = nobs
        self.head = (self.head + 1) % self.memory_size
        if self.head == 0:
            self.is_full = True

    def sample(self, batch_size):
        # length = self.memory_size if self.is_full else self.head + 1  #  why +1   ????
        length = self.memory_size if self.is_full else self.head
        idx = np.random.choice(np.arange(length), size=batch_size, replace=False)
        return np.array(self.obs_buf[idx]), \
               np.array(self.act_buf[idx]), \
               np.array(self.rwd_buf[idx]), \
               np.array(self.nobs_buf[idx])

class EpsilonExponentialDecay:
    def __init__(self, init, rate):
        self.init = init
        self.rate = rate

    def get(self, e):
        return self.init * (self.rate ** e)

class EpsilonLinearDecay:
    def __init__(self, init, end, episodes):
        self.init = init
        self.end = end
        self.episodes = episodes

    def get(self, e):
        return max(self.init - (self.init - self.end) / self.episodes * e, self.end)

class EpsilonExponentialConst:
    def __init__(self, init, rate, thr):
        self.init = init
        self.rate = rate
        self.thresh = thr

    def get(self, e):
        if self.init * (self.rate ** e)<self.thresh:
            return self.thresh
        else:
            return self.init * (self.rate ** e)