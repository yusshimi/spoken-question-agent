
from agents.sample01.tools.I2U.create_image_dataset import create_image_dataset


def create_dataset(task_name, base_data_dir, ):


    create_image_dataset(base_dir, items, train_images_per_folder=30, test_images_per_folder=10, overwrite=False)


class Trainer:
    def __init__(self):
        pass

    def pretrain(self, agent):
        self.train_i2u()
        self.train_u2s()
    
    def train_rl(self):
        self.train_rl()

    def train_i2u(self):
        pass

    def train_u2s(self):
        pass

    def train_rl(self):
        pass

def trainer():
    train_i2u()
    train_u2s()
    pass

def train_i2u():

    pass

def train_u2s():
    pass

def train_rl():
    pass