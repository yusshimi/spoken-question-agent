import librosa
import gymnasium as gym
from typing import Tuple
import numpy as np
import torch
from torchvision import transforms
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import os

from ..register_functions import make_env_custom_wrap


# class TransformImageWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)

#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224), antialias=True),  # リサイズを追加
#             transforms.Normalize(mean=self.mean, std=self.std)
#         ])

#         self.observation_space = gym.spaces.Dict({
#             'state': gym.spaces.Box(low=-10000, high=10000, shape=(3,)),
#             'leftimage': gym.spaces.Box(low=0, high=1, shape=(3, 224, 224)),
#             'rightimage': gym.spaces.Box(low=0, high=1, shape=(3, 224, 224)),
#         })
    
#     def transform_image(self, image: np.ndarray) -> np.ndarray:
#         image = image.transpose(2, 0, 1)
#         image = torch.FloatTensor(image / 255.0)
#         image = self.transform(image)
#         image = image.numpy()
#         return image
    
#     def reset(self, seed=None, options=None) -> dict:
#         observation, info = self.env.reset()
#         observation['leftimage'] = self.transform_image(observation['leftimage'])
#         observation['rightimage'] = self.transform_image(observation['rightimage'])
#         return observation, info
    
#     def step(self, action_waveform: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
#         observation, reward, terminated, truncated, info = self.env.step(action_waveform)
#         observation['leftimage'] = self.transform_image(observation['leftimage'])
#         observation['rightimage'] = self.transform_image(observation['rightimage'])
#         return observation, reward, terminated, truncated, info




# def wrap_task_env(base_env, **kwargs):
#     """
#     wrapperを適用した環境を返す関数
#     wrapperを追加するときはこの関数に必ず追加する
#     """
#     wrapped_env = TransformImageWrapper(base_env)
#     return wrapped_env

class TransformImageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),  # リサイズを追加
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # HuBERTモデルとプロセッサのロード
        # "facebook/hubert-base-ls960" が一般的です
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "facebook/hubert-base-ls960"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.audio_encoder = HubertModel.from_pretrained(self.model_name).to(self.device)
        self.audio_encoder.eval() # 推論モードに固定

        # リサンプラー (環境が22050Hz、HuBERTが16000Hzを想定)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=22050,
            new_freq=16000
        )

        audio_dim = self.audio_encoder.config.hidden_size

        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(low=-10000, high=10000, shape=(3,)),
            'leftimage': gym.spaces.Box(low=0, high=1, shape=(3, 224, 224)),
            # 'rightimage': gym.spaces.Box(low=0, high=1, shape=(3, 224, 224)),
            'audio': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(audio_dim,))
        })
    
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image / 255.0)
        image = self.transform(image)
        image = image.numpy()
        return image
    
    def extract_audio_features(self, wav_np: np.ndarray) -> np.ndarray:
        # 1. リサンプリング (22.05k -> 16k)
        wav_tensor = torch.from_numpy(wav_np).float()
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        wav_16k = self.resampler(wav_tensor).squeeze(0)

        # 2. 特徴抽出器による正規化
        inputs = self.feature_extractor(wav_16k, sampling_rate=16000, return_tensors="pt").to(self.device)

        # 3. 推論 (勾配計算なし)
        with torch.no_grad():
            outputs = self.audio_encoder(**inputs)
            # outputs.last_hidden_state は [batch, sequence, 768]
            # 時間方向に平均して1本のベクトル (768,) に変換
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy().flatten()

    
    def reset(self, seed=None, options=None) -> dict:
        observation, info = self.env.reset()
        observation['leftimage'] = self.transform_image(observation['leftimage'])
        # observation['rightimage'] = self.transform_image(observation['rightimage'])
        observation['audio'] = self.extract_audio_features(observation['audio'])
        return observation, info
    
    def step(self, action_waveform: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action_waveform)
        observation['leftimage'] = self.transform_image(observation['leftimage'])
        # # observation['rightimage'] = self.transform_image(observation['rightimage'])
        observation['audio'] = self.extract_audio_features(observation['audio'])
        return observation, reward, terminated, truncated, info




def wrap_task_env(base_env, **kwargs):
    """
    wrapperを適用した環境を返す関数
    wrapperを追加するときはこの関数に必ず追加する
    """
    wrapped_env = TransformImageWrapper(base_env)
    return wrapped_env