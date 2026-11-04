from typing import Callable
from pathlib import Path
import json
import os
import re
import glob

# from .tools.I2U.train_i2u import main as train_i2u_main
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from agents.base_agent_with_units import BaseAgentWithUnits
from agents.base_trainer import BaseTrainerWithUnits

from .tools.I2U.create_input_files2 import create_i2u_dataset
from .tools.I2U.train_i2u2 import train_i2u as train_i2u_

from .tools.U2S.preprocess2 import generate_u2s_dataset
from .tools.U2S.train2 import create_hparams, train_u2s as train_u2s_

from .tools.hifi_gan.env import AttrDict

from .utils.callbacks import ActionNoiseDecayCallback

# from food_task.py
class Action2SpeechWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action2speach: Callable[[np.ndarray], np.ndarray]):
        super().__init__(env)
        self.action2speach = action2speach
        self.action_space = self.env.action_space

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action_waveform = self.action2speach(action)
        observation, reward, terminated, truncated, info = self.env.step(action_waveform)
        # info['action_waveform'] = action_waveform
        return observation, reward, terminated, truncated, info


def get_latest_u2m_model_path(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_*"))
    if not checkpoints:
        return None
        
    # iterationの数値を抽出して最大値を持つファイルを見つける
    latest = max(checkpoints, key=lambda x: int(re.search(r'checkpoint_(\d+)', x).group(1)))
    return latest

class Trainer(BaseTrainerWithUnits):
    def __init__(self):
        pass

    def pretrain(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        # I2Uのモデルを訓練する
        i2u_info = self.train_i2u(
            agent=agent,
            image_paths=image_paths,
            audio_paths=audio_paths,
            base_config=base_config,
            pretrain_config=pretrain_config,
            agent_dir=agent_dir,
            agent_experiment_dir=agent_experiment_dir,
        )

        # U2Sのモデルを訓練する
        u2s_info = self.train_u2s(
            agent=agent,
            image_paths=image_paths,
            audio_paths=audio_paths,
            base_config=base_config,
            pretrain_config=pretrain_config,
            agent_dir=agent_dir,
            agent_experiment_dir=agent_experiment_dir,
        )

        info = {
            "i2u": {
                "word_map_path": i2u_info["word_map_path"],
                "i2u_model_path": i2u_info["output_model_path"],
            },
            "u2s": u2s_info,
        }
        return info


    def generate_i2u_dataset(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_experiment_dir: Path,
    ):
        # I2Uのデータセットを生成する
        i2u_data_dir = agent_experiment_dir / "data" / "I2U"
        data_info = create_i2u_dataset(
            agent,
            task_name=base_config['task_name'],
            image_paths=image_paths,
            audio_paths=audio_paths,
            output_data_dir=i2u_data_dir,
            captions_per_image=pretrain_config['I2U']['dataset']['captions_per_image'],
            min_word_freq=pretrain_config['I2U']['dataset']['min_word_freq'],
            max_len=pretrain_config['I2U']['dataset']['max_len'],
        )
        # ※ ここまで動作確認済み。これを、train_i2uの中に入れて、word_map_pathとかを返すように改良する予定

        return data_info
        
    
    def train_i2u(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        data_info = self.generate_i2u_dataset(
            agent,
            image_paths=image_paths,
            audio_paths=audio_paths,
            base_config=base_config,
            pretrain_config=pretrain_config,
            agent_experiment_dir=agent_experiment_dir,
        )

        i2u_pretrain_config = pretrain_config["I2U"]
        # ※ 実装中。generate_i2u_datasetをこの中で呼び出し、word_map_pathとかを受け取って、i2uを訓練できるようにする。
        word_map_path = data_info["word_map_path"]
        with open(word_map_path) as j:
            word_map = json.load(j)
        
        agent.i2u.init_model(word_map=word_map)

        i2u_data_dir = data_info["i2u_data_dir"]

        output_model_dir = agent_experiment_dir / "models" / "I2U"
        output_model_dir.mkdir(parents=True, exist_ok=True)

        output_model_path = output_model_dir / "i2u_best_model.pth"

        if i2u_pretrain_config["train_i2u"]:
            train_i2u_(
                model=agent.i2u,
                i2u_pretrain_config=i2u_pretrain_config, 
                input_data_folder=i2u_data_dir,
                data_filename=data_info["data_filename"],
                output_model_path=output_model_path,
                word_map=word_map,
            )
        return {
            "word_map_path": word_map_path, 
            "output_model_path": output_model_path,
        }
    
    def generate_u2m_dataset(
        self,
        agent: BaseAgentWithUnits,
        audio_paths,
        base_config,
        u2m_pretrain_config,
        agent_experiment_dir: Path,
    ):
        u2s_pretrain_data_dir = agent_experiment_dir / "data" / "U2S"
        data_info = generate_u2s_dataset(
            u2m_pretrain_config, 
            audio_paths=audio_paths,
            agent=agent,
            output_data_dir=u2s_pretrain_data_dir,
        )
        return data_info
            
    
    def train_u2s(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        # U2Sのデータセットを生成する
        u2m_pretrain_config = pretrain_config["U2S"]["u2m"]
        data_info = self.generate_u2m_dataset(
            agent=agent,
            audio_paths=audio_paths,
            base_config=base_config,
            u2m_pretrain_config=u2m_pretrain_config,
            agent_experiment_dir=agent_experiment_dir,
        )

        # U2Mのモデルを訓練する
        hparams = create_hparams(u2m_pretrain_config, data_file_paths=data_info)
        agent.u2s.init_u2m(hparams=hparams)

        u2m_model_dir = agent_experiment_dir / "models" / "U2S"
        train_u2m = u2m_pretrain_config["train_u2m"]

        if train_u2m:
            train_u2s_(
                model=agent.u2s.u2m, 
                u2s_pretrain_config=u2m_pretrain_config, 
                model_output_dir=u2m_model_dir,
                hparams=hparams,
            )

        m2s_pretrain_config = pretrain_config["U2S"]["m2s"]
        hparams = AttrDict(m2s_pretrain_config)
        agent.u2s.m2s.init_model(hparams=hparams)

        
        return {
            "u2m": {
                "u2m_model_dir": u2m_model_dir,
                "u2m_pretrain_config": u2m_pretrain_config,
                "data_info": data_info,
            },
            "m2s": {
                "m2s_pretrain_config": hparams,
            }
        }
    
    def env_wrapper(self, env, action2speech: Callable[[np.ndarray], np.ndarray]):
        return NotImplementedError()
    

        return agent

    
    def train_rl(
        self,
        agent: BaseAgentWithUnits,
        env_train,
        env_test,
        base_config,
        env_config,
        agent_config,
        pretrain_info,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        i2u_init_info = {
            "word_map_path": pretrain_info["i2u"]["word_map_path"],
        }
        i2u_model_path = pretrain_info["i2u"]["i2u_model_path"]

        u2m_init_info = {
            "config": pretrain_info["u2s"]["u2m"]["u2m_pretrain_config"],
            "data_file_paths": pretrain_info["u2s"]["u2m"]["data_info"],
        }
        u2m_model_path = get_latest_u2m_model_path(pretrain_info["u2s"]["u2m"]["u2m_model_dir"])

        m2s_init_info = {
            "config": pretrain_info["u2s"]["m2s"]["m2s_pretrain_config"],
            "device": agent_config["U2S"]["m2s"]["device"],
        }
        m2s_model_path = agent_dir / "models" / "U2S" / "m2s" / agent_config["U2S"]["m2s"]["model_file_name"]


        # 事前学習済みのモデルを読み込む
        agent.load(
            i2u_init_info=i2u_init_info,
            i2u_model_path=i2u_model_path,
            s2u_init_info=None,
            s2u_model_path=None,
            u2m_init_info=u2m_init_info,
            u2m_model_path=u2m_model_path,
            m2s_init_info=m2s_init_info,
            m2s_model_path=m2s_model_path,
        )

        # RLのモデルを初期化する
        word_map_path = pretrain_info["i2u"]["word_map_path"]
        with open(word_map_path) as j:
            word_map = json.load(j)

        agent.init_rl(
            word_map=word_map,
            device=agent_config["device"],
            special_words=agent_config["U2S"]["m2s"]["special_words"],
            beam_size=agent_config["I2U"]["beam_size"],
        )


        rl_model_config = agent_config["RL"]
        rl_model_config["i2u_d_embed"] = agent_config["I2U"]["d_embed"]
        rl_model_config["tensorboard_log_dir"] = agent_experiment_dir / rl_model_config["tensorboard_log"]

        # 環境をラップする
        env_train = agent.get_rl_env(env_train, rl_model_config)
        env_test = agent.get_rl_env(env_test, rl_model_config)

        agent.to(agent_config["device"])

        # RLモデルを取得する
        model = agent.get_rl_model(rl_model_config, env_train)

        eval_callback = EvalCallback(
            eval_env=env_test,
            eval_freq=env_config["eval_freq"],  # configで指定された頻度で評価
            n_eval_episodes=env_config["n_eval_episodes"],  # configで指定されたエピソード数で評価
            log_path=agent_experiment_dir / "models" / env_config["eval_log_path"],  # ログの保存先
            best_model_save_path=agent_experiment_dir / "models" / env_config["eval_log_path"]/ "best_model",  # ベストモデルの保存先
            deterministic=env_config["deterministic"],  # 評価時は決定論的な行動を使用
            render=env_config["render"]  # レンダリングはオフ
        )

        initial_sigma = agent_config['RL']['action_noise_sigma'] 
        total_steps = env_config['total_timesteps']
        
        # コールバックのインスタンス化
        noise_decay_callback = ActionNoiseDecayCallback(
            initial_sigma=initial_sigma,
            final_sigma=agent_config['RL'].get('action_noise_final_sigma', 0.1),
            decay_steps=agent_config['RL'].get('action_noise_decay_steps', int(total_steps * 0.5)),
        )

        # 修正2: eval_callback と noise_decay_callback をリストにまとめる
        # Stable Baselines3 は複数のコールバックをリスト形式で受け取れます
        callbacks = [eval_callback, noise_decay_callback]

        model.learn(
            total_timesteps=env_config["total_timesteps"],
            callback=callbacks,  # 修正: まとめたリストを渡す
            tb_log_name=agent_config["name"],
            reset_num_timesteps=True,
            progress_bar=False,
        )
        
