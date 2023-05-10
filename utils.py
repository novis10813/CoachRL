import csv
import os
from datetime import datetime


class CsvWriter:
    def __init__(self, file_name, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S-")
        self.path = os.path.join(file_path, current_time+file_name)
        with open(f'{self.path}.csv', mode='w', newline='') as csv_file:
            self.csv_writer = csv.writer(csv_file)
            self.csv_writer.writerow(['episode', 'reward', 'loss'])

    def log(self, episode, reward, loss):
        with open(f'{self.path}.csv', mode='a', newline='') as csv_file:
            self.csv_writer = csv.writer(csv_file)
            self.csv_writer.writerow([episode, reward, loss])
