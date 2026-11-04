import logging
from typing import Tuple, Optional, Type, List, Any, Callable
import numpy as np
import gymnasium as gym

from .dialogue_simulator import DialogueSimulator
from .data_models import DialogueFeedback, DialogueScene, EnvironmentState

class AgentStateManager(gym.Env):
    dialogue_simulator: DialogueSimulator
    initial_internal_state_function: Callable[[], np.ndarray]
    observation_space: gym.Space
    dialogue_scene: DialogueScene
    dialogue_feedback: Optional[DialogueFeedback]
    current_internal_state: np.ndarray
    current_env: EnvironmentState
    previous_env: EnvironmentState
    max_step: int
    talk_step: int

    def __init__(
        self,
        dialogue_simulator: DialogueSimulator,
        initial_internal_state_function: Optional[Callable[[], np.ndarray]] = None,
        max_step: int = 100,
    ):
        super().__init__()
        self.dialogue_simulator = dialogue_simulator
        self.initial_internal_state_function = initial_internal_state_function if initial_internal_state_function is not None else lambda: np.random.rand(3).astype(np.float32)
        self.max_step = max_step
        self.action_space = VariableLengthVectorSpace(dtype=np.float32, min_length=1)

        # 観察空間は内部状態のみ
        self.observation_space = gym.spaces.Dict({
            "internal_state": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32), 
})

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        logging.info("Resetting environment to initial state.")
        self.dialogue_simulator.current_scene = self.dialogue_simulator._initialize_current_scene()
        self.dialogue_scene = self.dialogue_simulator.get_current_scene_info()
        self.dialogue_feedback = None
        self.current_internal_state = self.initial_internal_state_function()  # リセット時に内部状態を初期化
        self.current_env = EnvironmentState(self.dialogue_scene, self.dialogue_feedback, self.current_internal_state)
        self.talk_step = 0
        obs = {"internal_state": self.current_env.internal_state}
        return obs, {}

    def step(self, action_waveform: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        logging.info("Processing action and updating environment.")
        self.previous_env = self.current_env

        # DialogueFeedbackを取得
        self.dialogue_feedback = self.dialogue_simulator.get_dialogue_feedback(action_waveform)

        # 次のDialogueSceneを取得
        self.dialogue_scene = self.dialogue_simulator.get_current_scene_info()

        # 次の環境状態の生成
        self.current_env = EnvironmentState(self.dialogue_scene, self.dialogue_feedback, self.current_internal_state)

        # 報酬と終了の初期値(Wrapperで上書き前提)
        reward = 0.0
        terminated = False
        
        self.talk_step += 1
        truncated = self.talk_step >= self.max_step

        #初期値は内部状態のみを観測とする(ObservationProcessingWrapperでself.current_envから設定可能)
        obs = {"internal_state": self.current_env.internal_state}

        return obs, reward, terminated, truncated, {}

class VariableLengthVectorSpace(gym.Space):
    dtype: Type[Any]
    min_length: int
    max_length: Optional[int]

    def __init__(self, dtype=np.float32, min_length: int = 1, max_length: Optional[int] = None):
        """
        可変長ベクトルを表すカスタムSpaceクラス。

        Args:
            dtype (type, optional): データ型。デフォルトは np.float32。
            min_length (int, optional): ベクトルの最小長。デフォルトは 1。
            max_length (Optional[int], optional): ベクトルの最大長。Noneの場合、制限なし。
        """
        # GymnasiumのSpaceクラスはshapeとdtypeを必要とします。
        # 可変長のためshapeはNoneまたは空のタプルに設定します。
        super().__init__(shape=None, dtype=dtype)
        self.dtype = dtype
        self.min_length = min_length
        self.max_length = max_length  # Noneの場合、制限なし

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """
        可変長のため、shapeはNoneを返す。
        必要に応じて最小長を基に形状情報を提供することも可能。
        """
        return None  # 可変長のため固定のshapeは存在しない

    def sample(self) -> np.ndarray:
        """
        ランダムなサンプルを生成。
        """
        if self.max_length is not None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        #最大長がない場合は100より小さい範囲に設定(実際にはこの関数を使わずエージェント側で代わりとなるsample関数を作る必要がある)
        else:
            length = np.random.randint(self.min_length, 100)
        return self.dtype(np.random.randn(length))

    def contains(self, x) -> bool:
        """
        サンプルがスペースに含まれるかどうかを判定。

        Args:
            x: チェック対象のサンプル。

        Returns:
            bool: 含まれる場合はTrue、そうでない場合はFalse。
        """
        if not isinstance(x, (list, np.ndarray)):
            return False
        length = len(x)
        if length < self.min_length:
            return False
        if self.max_length is not None and length > self.max_length:
            return False
        if not np.issubdtype(np.array(x).dtype, np.dtype(self.dtype).type):
            return False
        return True

    def to_jsonable(self, sample_n: List[np.ndarray]) -> List[List[float]]:
        """
        サンプルをJSON形式に変換。

        Args:
            sample_n (List): サンプルのリスト。

        Returns:
            List[List[float]]: JSON互換のリスト。
        """
        return [x.tolist() for x in sample_n]

    def from_jsonable(self, sample_n: List[List[float]]) -> List[np.ndarray]:
        """
        JSON形式からサンプルに変換。

        Args:
            sample_n (List[List[float]]): JSON形式のサンプルリスト。

        Returns:
            List[np.ndarray]: NumPy配列のリスト。
        """
        return [self.dtype(x) for x in sample_n]

    def __repr__(self):
        return f"VariableLengthVectorSpace(dtype={self.dtype}, min_length={self.min_length}, max_length={self.max_length})"
