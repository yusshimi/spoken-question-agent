import tracemalloc
from typing import Callable

import gymnasium as gym
import numpy as np
import psutil
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch

from agents.base_agent import BaseAgent
from abc import ABC, abstractmethod



class BaseTrainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def pretrain(self, agent: BaseAgent, audio_data_dir, image_data_dir, config):
        return NotImplementedError()
    
    @abstractmethod
    def train_rl(self, agent: BaseAgent, audio_data_dir, image_data_dir, config):
        return NotImplementedError()



class BaseTrainerWithUnits(BaseTrainer):
    def __init__(self):
        super().__init__()

    def pretrain(self, agent: BaseAgent, audio_data_dir, image_data_dir, config) -> BaseAgent:
        agent = self.train_i2u(agent, config)
        agent = self.train_u2s(agent, config)
        return agent
    
    @abstractmethod
    def train_i2u(self, agent:BaseAgent, config) -> BaseAgent:
        return NotImplementedError()

    @abstractmethod
    def train_u2s(self, agent:BaseAgent, config) -> BaseAgent:
        return NotImplementedError()
    
    @abstractmethod
    def env_wrapper(self, env, action2speech: Callable[[np.ndarray], np.ndarray]):
        return NotImplementedError()
    