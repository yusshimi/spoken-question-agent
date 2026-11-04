import logging
import json
import os
from typing import Any, Optional
import numpy as np
from PIL import Image
import soundfile as sf
from collections import deque

from .data_models import DialogueScene, DialogueFeedback, EnvironmentState, SavedInteraction, Item, FeedbackItem

class DataLogger:
    def __init__(self, images_dir: str = "./images", audio_dir: str = "./audio", log_dir: str = "./logs", sampling_rate: int = 22050):
        self.images_dir: str = images_dir
        self.audio_dir: str = audio_dir
        self.log_dir: str = log_dir
        self.sampling_rate = sampling_rate
        self.item_dict: dict[int, Item] = {}
        self.saved_interaction_counter: dict[str, int] = {"train": 0, "test": 0}
        self.scene_image_queues: dict[str, deque[list[Optional[str]]]] = {"train": deque(maxlen=2), "test": deque(maxlen=2)}
        # データ保存用ディレクトリを作成
        os.makedirs(os.path.join(self.images_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.images_dir, "test"), exist_ok=True)
        os.makedirs(os.path.join(self.audio_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.audio_dir, "test"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "test"), exist_ok=True)

        # SavedInteractionインスタンス保存用リストを作成
        self.saved_interaction_lists: dict[str, list[SavedInteraction]] = {"train": [], "test": []}

    def log_interaction(self, current_env: EnvironmentState, action_waveform: np.ndarray, next_env: EnvironmentState, reward: float, mode: str = "train", dialogue_counter: int = 1):
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        os.makedirs(os.path.join(self.audio_dir, mode, str(dialogue_counter)), exist_ok=True)

        self.saved_interaction_counter[mode] += 1

        # EnvironmentStateインスタンスを辞書に変換
        current_env_dict = self._encode_environment_state(current_env, True, mode=mode, dialogue_counter=dialogue_counter)
        next_env_dict = self._encode_environment_state(next_env, False, mode=mode, dialogue_counter=dialogue_counter)
        
        # action_waveformをWAVファイルとして保存し、ファイルパスを取得
        action_waveform_path = self.save_audio(action_waveform, f"{self.saved_interaction_counter[mode]}_action-waveform", mode, dialogue_counter)

        # SavedInteractionインスタンスを作成し、saved_interaction_listsに追加
        saved_interaction = SavedInteraction(current_env_dict, action_waveform_path, next_env_dict, reward)
        self.saved_interaction_lists[mode].append(saved_interaction)
    
    def load_interaction(self, saved_interaction: SavedInteraction) -> tuple[EnvironmentState, np.ndarray, EnvironmentState, float]:
        # saved_interactionの情報を抽出
        current_env_dict = saved_interaction.environment_state_dict
        action_waveform_path = saved_interaction.action_waveform_path
        next_env_dict = saved_interaction.next_environment_state_dict
        reward = saved_interaction.reward

        # 辞書からEnvironmentStateインスタンスに変換
        current_env = self._decode_environment_state(current_env_dict)
        next_env = self._decode_environment_state(next_env_dict)

        # action_waveformをWAVファイルから取得
        action_waveform, sampling_rate = sf.read(action_waveform_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Sampling rate of '{action_waveform_path}' must be {self.sampling_rate}")

        return current_env, action_waveform, next_env, reward

    def save_image(self, image: np.ndarray, item: Item, image_index: int = 0, mode: str = "train"):
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = os.path.join(self.images_dir, mode)
        # PIL Imageに変換して保存
        pil_img = Image.fromarray(image.astype('uint8'))  # データ型をuint8に変換
        # image_path = os.path.join(self.images_dir, f"{item.id}_{item.name}.png")
        image_path = os.path.join(sub_dir, f"{item.id}_{item.name}_{image_index}.png")
        pil_img.save(image_path)
        logging.info(f"Image for '{item.name}' (index={image_index}) saved at '{image_path}'.")
    
    def load_image(self, item: Item, image_index: int = 0, mode: str = 'train') -> np.ndarray:
        # 画像を読み込み
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = os.path.join(self.images_dir, mode)
        image_path = os.path.join(sub_dir, f"{item.id}_{item.name}_{image_index}.png")
        if not os.path.exists(image_path):
            logging.error(f"Image for '{item.name}' not found at '{image_path}'.")
            return None
        pil_img = Image.open(image_path)
        image = np.array(pil_img)
        logging.info(f"Image for '{item.name}' (index={image_index}) loaded from '{image_path}'.")
        return image
    
    def save_audio(self, waveform: np.ndarray, file_name: str, mode: str = "train", dialogue_counter: int = 1) -> str:
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = os.path.join(self.audio_dir, mode, str(dialogue_counter))
        audio_path = os.path.join(sub_dir, f"{file_name}.wav")
        sampling_rate = self.sampling_rate
        sf.write(audio_path, waveform, sampling_rate)
        logging.info(f"Audio saved at '{audio_path}'.")

        return audio_path

    def save_logfile(self, file_name: str, mode: str = "train"):
        # saved_interaction_listsの各SavedInteractionインスタンスを辞書に変換
        saved_interaction_dict_list = []
        for saved_interaction in self.saved_interaction_lists[mode]:
            saved_interaction_dict_list.append({
                "environment_state": saved_interaction.environment_state_dict,
                "action_waveform_path": saved_interaction.action_waveform_path,
                "next_environment_state": saved_interaction.next_environment_state_dict,
                "reward": saved_interaction.reward
            })
        
        # saved_interaction_dict_listをファイルとして保存
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = os.path.join(self.log_dir, mode)
        log_path = os.path.join(sub_dir, f"{file_name}.json")
        with open(log_path, "w", encoding='utf-8') as file:
            json.dump(saved_interaction_dict_list, file, indent=2, ensure_ascii=False)
        logging.info(f"Interaction log saved at '{log_path}'.")
        #リスト、カウンターの初期化
        self.saved_interaction_lists[mode] = []
        self.saved_interaction_counter[mode] = 0
    
    def load_logfile(self, file_name: str, mode: str = "train") -> list[SavedInteraction]:
        # ログファイルからSavedInteractionインスタンスの情報を辞書として読み込む
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = os.path.join(self.log_dir, mode)
        log_path = os.path.join(sub_dir, f"{file_name}.json")
        if not os.path.exists(log_path):
            logging.error(f"logfile not found at '{log_path}'.")
            return None
        with open(log_path, "r", encoding='utf-8') as file:
            saved_interaction_dict_list = json.load(file)
        logging.info(f"Interaction log loaded from '{log_path}'.")

        # 辞書からSavedInteractionインスタンスに変換し、リストにまとめる
        saved_interaction_list = []
        for saved_interaction_dict in saved_interaction_dict_list:
            environment_state_dict = saved_interaction_dict["environment_state"]
            action_waveform_path = saved_interaction_dict["action_waveform_path"]
            next_environment_state_dict = saved_interaction_dict["next_environment_state"]
            reward = saved_interaction_dict["reward"]
            saved_interaction = SavedInteraction(environment_state_dict, action_waveform_path, next_environment_state_dict, reward)
            saved_interaction_list.append(saved_interaction)
        return saved_interaction_list
    
    def register_scene_paths(self, paths: list[Optional[str]], mode: str):
        self.scene_image_queues[mode].append(paths)

    def _encode_environment_state(self, environment_state: EnvironmentState, is_current_env: bool, mode: str = "train", dialogue_counter: int = 1) -> dict[str, Any]:
        if is_current_env:
            state_label = "current"
        else:
            state_label = "next"
        
        # DialogueSceneインスタンスを辞書に変換
        scene_id = environment_state.dialogue_scene.scene_id
        prompt_waveform_path = self.save_audio(environment_state.dialogue_scene.prompt_waveform, f"{self.saved_interaction_counter[mode]}_{state_label}-env_prompt-waveform", mode, dialogue_counter)  # prompt_waveformをWAVファイルとして保存し、ファイルパスを取得
        prompt_text = getattr(environment_state.dialogue_scene, "prompt_text", None)
        scene_image_queue = self.scene_image_queues[mode]
        if is_current_env:
            image_paths = scene_image_queue[0] if len(scene_image_queue) >= 1 else []
        else:
            image_paths = scene_image_queue[1] if len(scene_image_queue) >= 2 else []

        dialogue_scene_dict = {
            "scene_id": scene_id,
            "prompt_waveform_path": prompt_waveform_path,
            "prompt_text": prompt_text,
            "image_paths": image_paths
        }

        # DialogueFeedbackインスタンスを辞書に変換
        if environment_state.dialogue_feedback is not None:
            selected_item_attributes = None
            selected_item_image_path = None
            if environment_state.dialogue_feedback.selected_item is not None:
                selected_item_id = environment_state.dialogue_feedback.selected_item.id
                # アトリビュートを取得（sを忘れずに）
                selected_item_attributes = getattr(environment_state.dialogue_feedback.selected_item, "attributes", None)
                # 画像パスの特定ロジック
                if is_current_env:
                    # 前回の interaction の画像パスを参照
                    image_candidates = self.saved_interaction_lists[mode][-1].environment_state_dict["dialogue_scene"]["image_paths"] if self.saved_interaction_lists[mode] else []
                else:
                    image_candidates = self.scene_image_queues[mode][0]
                
                item_name = self.item_dict[selected_item_id].name
                matched_paths = [path for path in image_candidates if path and item_name in os.path.basename(path)]
                selected_item_image_path = matched_paths[0] if matched_paths else None
            
            response_waveform_path = self.save_audio(environment_state.dialogue_feedback.response_waveform, f"{self.saved_interaction_counter[mode]}_{state_label}-env_response-waveform", mode, dialogue_counter)  # response_waveformをWAVファイルとして保存し、ファイルパスを取得

            # --- 追加: LLM判定スコアの保存 ---
            response_text = getattr(environment_state.dialogue_feedback, "response_text", None)
            response_quality = getattr(environment_state.dialogue_feedback, "response_quality", None)
            evaluation_logic = getattr(environment_state.dialogue_feedback, "evaluation_logic", None)

            dialogue_feedback = {
                "selected_item_image_path": selected_item_image_path,
                "response_waveform_path": response_waveform_path,
                "selected_item_attributes": selected_item_attributes,
                "response_text": response_text,
                "evaluation_logic": evaluation_logic,
                "response_quality": response_quality
            }
        else:
            dialogue_feedback = None
        
        # # internal_stateを取得
        # internal_state = environment_state.internal_state.astype(str).tolist()

        # EnvironmentStateインスタンスを辞書に変換
        environment_state_dict = {
            "dialogue_scene": dialogue_scene_dict,
            "dialogue_feedback": dialogue_feedback,
            # "internal_state": internal_state
        }
        return environment_state_dict
    
    def _decode_environment_state(self, environment_state_dict: dict[str, Any], mode: str = "train") -> EnvironmentState:
        # DialogueSceneインスタンスを作成
        scene_id = environment_state_dict["dialogue_scene"]["scene_id"]

        prompt_waveform_path = environment_state_dict["dialogue_scene"]["prompt_waveform_path"]
        prompt_waveform, sampling_rate = sf.read(prompt_waveform_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Sampling rate of '{prompt_waveform_path}' must be {self.sampling_rate}")
        
        image_paths = environment_state_dict["dialogue_scene"]["image_paths"]
        images = []
        for path in image_paths:
            if path is None:
                # 保存されなかった画像はダミー画像を使う
                images.append(np.zeros((512, 512, 3), dtype=np.uint8))
            else:
                pil_img = Image.open(path)
                images.append(np.array(pil_img))
        
        dialogue_scene = DialogueScene(scene_id, prompt_waveform, images)
        dialogue_scene.prompt_text = environment_state_dict["dialogue_scene"].get("prompt_text", "")

        # DialogueFeedbackインスタンスを作成
        if environment_state_dict["dialogue_feedback"] is not None:
            if environment_state_dict["dialogue_feedback"]["selected_item_image_path"] is not None:
                selected_item_image_path = environment_state_dict["dialogue_feedback"]["selected_item_image_path"]
                selected_item_id = self.get_item_id_from_path(selected_item_image_path)
                selected_item = self.item_dict[selected_item_id]
                pil_img = Image.open(selected_item_image_path)
                feedback_item = FeedbackItem(selected_item.id, np.array(pil_img), selected_item.attributes)
            else:
                feedback_item = None
            
            response_waveform_path = environment_state_dict["dialogue_feedback"]["response_waveform_path"]
            response_waveform, sampling_rate = sf.read(response_waveform_path)
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"Sampling rate of '{response_waveform_path}' must be {self.sampling_rate}")
            
            dialogue_feedback = DialogueFeedback(feedback_item, response_waveform)
            # 追加: 判定品質の復元
            dialogue_feedback.response_text = environment_state_dict["dialogue_feedback"].get("response_text", "")
            dialogue_feedback.response_quality = environment_state_dict["dialogue_feedback"].get("response_quality", 0)
            dialogue_feedback.evaluation_logic = environment_state_dict["dialogue_feedback"].get("evaluation_logic", "") # 追加        
        else:
            dialogue_feedback = None
        
        # internal_stateを取得
        internal_state = np.array([], dtype=np.float32)

        # EnvironmentStateインスタンスを作成
        environment_state = EnvironmentState(dialogue_scene, dialogue_feedback, internal_state)
        return environment_state
    
    def _image_to_id(self, image: np.ndarray) -> int:
        for item in self.item_dict.values():
            if item.image is None:
                continue # Itemインスタンスに画像が設定されていない場合、以降の処理をスキップ

            if np.array_equal(image, item.image):
                return item.id # 画像に対応するItemインスタンスのidを返す
        
        raise ValueError("Image data not found.")
    
    def get_item_id_from_path(self, path: str) -> int:
        # ファイル名部分を取得
        filename = os.path.basename(path)
        # 拡張子を除く
        basename = os.path.splitext(filename)[0]
        # 最初のアンダースコアの位置を探す
        underscore_index = basename.find('_')
        if underscore_index == -1:
            raise ValueError(f"Filename format invalid (missing underscore): {filename}")
        
        item_id_str = basename[:underscore_index]
        if item_id_str.isdigit():
            return int(item_id_str)
        else:
            raise ValueError(f"Item ID is not numeric: {item_id_str}")