from stable_baselines3.common.callbacks import BaseCallback

class ActionNoiseDecayCallback(BaseCallback):
    def __init__(self, initial_sigma, final_sigma, decay_steps, verbose=0):
        super().__init__(verbose)
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        # 現在のステップ数に応じてノイズの強さ(sigma)を計算
        if self.num_timesteps < self.decay_steps:
            # 線形減少の計算式
            fraction = self.num_timesteps / self.decay_steps
            current_sigma = self.initial_sigma + fraction * (self.final_sigma - self.initial_sigma)
        else:
            current_sigma = self.final_sigma

        # モデルのaction_noiseの内部パラメータを直接書き換える
        if self.model.action_noise is not None:
            # NormalActionNoiseの場合は _sigma 属性を更新
            self.model.action_noise._sigma = current_sigma
            
        # TensorBoardで確認できるように記録
        self.logger.record("train/action_noise_sigma", current_sigma)
        return True