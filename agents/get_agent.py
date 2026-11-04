import os
import sys
import json
from pathlib import Path
import importlib


from agents.base_agent import BaseAgent
from agents.base_trainer import BaseTrainer



def get_agent_path(agent_name: str) -> dict[Path]:
    agent_dir = os.path.join(os.path.dirname(__file__), agent_name)
    agent_config_file = os.path.join(agent_dir, "agent_config.json")
    return {
        "agent_dir": agent_dir,
        "agent_config_file": agent_config_file
    }


def get_agent(agent_name:str, agent_config) -> tuple[BaseAgent, BaseTrainer]:

    agent_path = get_agent_path(agent_name)
    agent_dir = agent_path["agent_dir"]
    agent_config_file = agent_path["agent_config_file"]

    """
    エージェントの読み込み方法を悩み中
    1. 各フォルダに__init__.pyを作成して、
        AgentクラスとTrainerクラスをある固定の名前で作成してもらい、それを読み込む
        メリット：     pythonで標準的に使われている方法なので、他の人が見ても分かりやすい
        デメリット：   各フォルダに__init__.pyを作成する必要がある。
                        クラス名を固定する必要がある。
    2. 各フォルダに__init__.pyを作成して、
       AgentクラスとTrainerクラスを自由な名前で作成してもらい、そのクラス名をconfigで指定してもらう
        メリット：     pythonで標準的に使われている方法
                        クラス名を自由に決めることができ、複数のエージェントを定義しやすい
        デメリット：   configファイルを書くフォルダに作成する必要あり
    3. 各フォルダにget_agent関数とget_trainer関数を作成してもらい、それをsys.pathで読み込む
        メリット：     __init__.py が不要
                        クラス名を自由に決めることができ、複数のエージェントを定義しやすい
        デメリット：   sys.pathを変更する必要があり、並列処理やエラーメッセージの理解に問題が生じる可能性あり

    一旦、環境と同じように、2の方法で実装することにする
    """

    # エージェントディレクトリが存在するか確認
    if not os.path.isdir(agent_dir):
        raise ValueError(f"エージェント '{agent_name}' は存在しません")
    
    try:
        with open(agent_config_file, 'r') as f:
            agent_class_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"設定ファイル '{agent_config_file}' が見つかりません")
    except json.JSONDecodeError:
        raise ValueError(f"設定ファイル '{agent_config_file}' のJSONフォーマットが無効です")
    
    # 必要なキーが存在するか確認
    required_keys = ['agent_class', 'trainer_class']
    for key in required_keys:
        if key not in agent_class_config:
            raise ValueError(f"エージェント '{agent_name}' の設定に '{key}' が見つかりません")

    # try:
    # モジュールをインポート
    agent_module = importlib.import_module(f".{agent_name}.agent", package="agents")
    trainer_module = importlib.import_module(f".{agent_name}.trainer", package="agents")
    
    # クラスを取得
    agent_class = getattr(agent_module, agent_class_config['agent_class'])
    trainer_class = getattr(trainer_module, agent_class_config['trainer_class'])
    # 型チェック
    if not issubclass(agent_class, BaseAgent):
        raise TypeError(f"'{agent_class_config['agent_class']}' はBaseAgentを継承していません")
    
    if not issubclass(trainer_class, BaseTrainer):
        raise TypeError(f"'{agent_class_config['trainer_class']}' はBaseTrainerを継承していません")
    
    agent = agent_class(agent_config)
    trainer = trainer_class()
    return agent, trainer
    

    # except ImportError as e:
    #     raise ImportError(f"エージェント '{agent_name}' のモジュールをインポートできません: {e}")
    # except AttributeError as e:
    #     raise AttributeError(f"エージェント '{agent_name}' に必要なクラスが定義されていないか、configファイルの設定が間違っています: {e}")