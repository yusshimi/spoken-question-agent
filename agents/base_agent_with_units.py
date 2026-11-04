from abc import ABC, abstractmethod
from pathlib import Path
from agents.base_agent import BaseAgent


class BaseImage2Unit(ABC):
    def get_unit(self, path: Path) -> list[str]:
        # 画像からユニットを取得する処理を実装
        pass

class BaseSpeech2Unit(ABC):
    @abstractmethod
    def get_unit(self, paths: Path) -> list[str]:
        pass

class BaseUnit2Speech(ABC):
    def get_speech(self, unit: list[str]) -> str:
        # ユニットから音声を取得する処理を実装
        pass

class BaseRLUnit(ABC):
    def infer(self, unit: list[str]) -> str:
        # ユニットからRLの行動を取得する処理を実装
        pass


class BaseAgentWithUnits(BaseAgent, ABC):
    def __init__(self, agent_config):
        super().__init__(agent_config)
    
    
    @property
    @abstractmethod
    def i2u(self) -> BaseImage2Unit:
        """Image2Unitのインスタンスを返す抽象プロパティ"""
        pass
    
    @property
    @abstractmethod
    def s2u(self) -> BaseSpeech2Unit:
        """Speech2Unitのインスタンスを返す抽象プロパティ"""
        pass
    
    @property
    @abstractmethod
    def u2s(self) -> BaseUnit2Speech:
        """Unit2Speechのインスタンスを返す抽象プロパティ"""
        pass
    
    def pretrain(self):
        self.train_i2u()
        self.train_u2s()
        self.train_s2u()
    
    # I2Uモデルを訓練する場合、この関数をオーバーライドする
    def train_i2u(self):
        pass
    
    # U2Sモデルを訓練する場合、この関数をオーバーライドする
    def train_u2s(self):
        pass
    
    # S2Uモデルを訓練する場合、この関数をオーバーライドする
    def train_s2u(self):
        pass
