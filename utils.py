import csv
import os


class CsvWriter:
    def __init__(self, file_name, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        self.csv_file = open(f'{file_path}/{file_name}.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['episode', 'reward', 'loss'])

    def log(self, episode, reward, loss):
        self.csv_writer.writerow([episode, reward, loss])

    def close(self):
        self.csv_file.close()
