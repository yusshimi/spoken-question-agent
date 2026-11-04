from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.td3 import TD3
from stable_baselines3.td3.policies import MultiInputPolicy

from agents.original_progressbar_callback import ProgressBarCallback

class BaseAgent(ABC):
    def __init__(self, agent_config):
        super().__init__()

        self.agent_config = agent_config
    
    # エージェントにおいて事前学習が必要なら、この関数をオーバーライドしてください。
    @abstractmethod
    def pretrain(self):
        pass

    # 強化学習の行動空間を発話データに変換する関数
    @abstractmethod
    def action2speech(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

    # オリジナルのモデルを定義する場合は、この関数をオーバーライドしてください。
    def get_rl_model(self, env_train: gym.Env) -> BaseAlgorithm:
        return TD3(MultiInputPolicy, env_train)
    
    # オリジナルのEvalCallbackを定義する場合は、この関数をオーバーライドしてください。
    # (memo) 分かりづらいので、agent_configに初期値としていくらかのパラメータを置いておいて、
    # それを変更してもらえればCallbackに反映できるようにする方がだいぶ効率がいいような気がする
    # それなら BaseTrainer に渡されてる agent_config をそのまま使えるのでこの関数を消せる
    def get_eval_callback(self, env_test: gym.Env) -> EvalCallback | None:
        return None
    
    # オリジナルのProgressBarCallbackを定義する場合は、この関数をオーバーライドしてください。
    # (memo) こちらも分かりづらいので消したい。
    def get_progress_callback(self) -> ProgressBarCallback | None:
        return None
