import os
import importlib
from typing import Optional
import json

from pathlib import Path
from .register_functions import make_env_base


def get_task_path(task_name: str) -> dict[Path]:
    task_dir = os.path.join(os.path.dirname(__file__), task_name)

    task_config_file = os.path.join(task_dir, "task_config.json")
    with open(task_config_file, "r", encoding="utf-8") as file:
        task_config = json.load(file)
    task_config_name = task_config["task_config_name"]

    config_file = os.path.join(task_dir, task_config_name)

    return {
        "task_dir": task_dir,
        "config_file": config_file,
        "task_config_file": task_config_file
    }

def get_env(
    task_name,
    data_logger,
    llm,
    speech_recognizer,
    speech_synthesizer,
    image_generator,
    llm_prompt_name,
    mode="train",
    record_mode=False,
    num_images_per_item=1,
    sampling_rate=22050,
    enable_audio_response=True,
    **kwargs
):
    """
    指定されたタスク名に基づいて環境を作成する関数
    
    Parameters:
    -----------
    task_name : str
        タスクの名前（例: "food_task"）
    data_logger : DataLogger
        データロガーインスタンス
    llm : LLMComponent
        LLMインスタンス
    speech_recognizer : SpeechRecognizerComponent
        音声認識インスタンス
    speech_synthesizer : SpeechSynthesizerComponent
        音声合成インスタンス
    image_generator : ImageGeneratorComponent
        画像生成インスタンス
    llm_prompt_name : str
        LLMが使用するプロンプト
    mode : str
        "train" または "test"
    record_mode : bool
        対話を記録するかどうか
    num_images_per_item : int
        アイテムごとの画像数
    sampling_rate : int
        音声のサンプリングレート
    enable_audio_response : bool
        音声返答を有効にするか
    
    Returns:
    --------
    env : gym.Env
        環境インスタンス
    """
    # 必須パラメータのチェック
    if data_logger is None:
        raise ValueError("data_logger is required")
    if llm is None:
        raise ValueError("llm is required")
    if speech_recognizer is None:
        raise ValueError("speech_recognizer is required")
    if speech_synthesizer is None:
        raise ValueError("speech_synthesizer is required")
    if image_generator is None:
        raise ValueError("image_generator is required")
    if llm_prompt_name is None:
        raise ValueError("llm_prompt_name is required")
    
    # タスク設定ファイルのパスを取得
    task_path = get_task_path(task_name)
    task_dir = task_path["task_dir"]
    config_file = task_path["config_file"]
    task_config_file = task_path["task_config_file"]
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"タスク設定ファイルが見つかりません: {config_file}")
    
    # task_config.jsonからmax_stepを取得
    max_step = 100  # デフォルト値
    if os.path.exists(task_config_file):
        with open(task_config_file, 'r') as f:
            task_config = json.load(f)
            max_step = task_config.get("max_step", max_step)
    
    # タスク固有の関数をインポート
    try:
        task_functions = importlib.import_module(f".{task_name}.task_functions", package="tasks")
        reward_function = getattr(task_functions, "reward_function", None)
        termination_function = getattr(task_functions, "termination_function", None)
        truncation_function = getattr(task_functions, "truncation_function", None)
        internal_state_update_function = getattr(task_functions, "internal_state_update_function", None)
        observation_function = getattr(task_functions, "observation_function", None)
        initial_internal_state_function = getattr(task_functions, "initial_internal_state_function", None)
    except ImportError as e:
        raise ImportError(f"タスク関数のインポートに失敗しました: {e}")
    
    # make_env_baseを使用して基本環境を作成
    env = make_env_base(
        data_logger=data_logger,
        llm=llm,
        speech_recognizer=speech_recognizer,
        speech_synthesizer=speech_synthesizer,
        image_generator=image_generator,
        config_path=config_file,
        llm_prompt_name=llm_prompt_name,
        mode=mode,
        num_images_per_item=num_images_per_item,
        sampling_rate=sampling_rate,
        enable_audio_response = enable_audio_response,
        initial_internal_state_function=initial_internal_state_function,
        max_step=max_step,
        reward_function=reward_function,
        termination_function=termination_function,
        truncation_function=truncation_function,
        internal_state_update_function=internal_state_update_function,
        observation_function=observation_function,
        record_mode=record_mode
    )
    
    # タスク固有のラッパーがあれば適用
    try:
        task_wrappers = importlib.import_module(f".{task_name}.task_wrappers", package="tasks")
        wrap_task_env = getattr(task_wrappers, "wrap_task_env", None)
        if wrap_task_env:
            env = wrap_task_env(env, **kwargs)
    except (ImportError, AttributeError):
        # タスク固有のラッパーがない場合は無視
        pass
    
    return env