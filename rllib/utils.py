import csv
import torch
import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class CsvWriter:
    def __init__(self, file_name, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S-")
        self.path = os.path.join(file_path, current_time + file_name)
        with open(f"{self.path}.csv", mode="w", newline="") as csv_file:
            self.csv_writer = csv.writer(csv_file)
            self.csv_writer.writerow(["episode", "reward", "loss"])

    def log(self, episode, reward, loss):
        with open(f"{self.path}.csv", mode="a", newline="") as csv_file:
            self.csv_writer = csv.writer(csv_file)
            self.csv_writer.writerow([episode, reward, loss])


class StateProcessor:
    def __init__(self, device):
        self.device = device

    def to_numpy(self, state):
        return state.cpu().numpy()

    def to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)


class LinearAnneal:
    def __init__(self, start_eps=0.9, end_eps=1e-4, episodes=1000):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_rate = (start_eps - end_eps) / episodes

    def anneal(self):
        if self.start_eps > self.end_eps:
            self.start_eps -= self.decay_rate
        return self.start_eps


class Plotter:
    def __init__(self):
        self.episodes_durations = []

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episodes_durations, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not self.show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
