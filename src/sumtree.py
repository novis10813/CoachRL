import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.nodes = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.capacity = capacity
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.capacity - 1
        change = value - self.nodes[data_idx]
        self.nodes[idx] = value
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.capacity
        self.real_size = min(self.capacity, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total
        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]
        data_idx = idx - self.capacity + 1
        return data_idx, self.nodes[idx], self.data[data_idx]
