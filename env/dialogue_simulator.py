import yaml
import logging
import random
import json
from typing import Dict, Any, Optional, List
import numpy as np
import re
import os

from .generative_models import LLMComponent, SpeechRecognizerComponent, SpeechSynthesizerComponent, ImageGeneratorComponent
from .data_logger import DataLogger
from .data_models import DialogueScene, DialogueFeedback, Item, FeedbackItem

class DialogueSimulator:
    def __init__(
        self, 
        config_path: str, 
        llm: LLMComponent, 
        speech_recognizer: SpeechRecognizerComponent,
        speech_synthesizer: SpeechSynthesizerComponent,
        image_generator: ImageGeneratorComponent, 
        data_logger: DataLogger,
        llm_prompt_name: str,
        mode: str = "train",
        num_images_per_item: int = 10,
        sampling_rate: int = 22050,
        enable_audio_response: bool = True
    ):
        self.llm = llm
        self.speech_recognizer = speech_recognizer
        self.speech_synthesizer = speech_synthesizer
        self.image_generator = image_generator
        self.data_logger = data_logger
        self.mode = mode
        self.num_images_per_item = num_images_per_item
        self.sampling_rate = sampling_rate
        self.enable_audio_response = enable_audio_response
        self.last_question_text = ""

        self.load_config(config_path)
        self.data_logger.item_dict = self._load_item_dict_from_config()
        self.prompt_template = self.load_llm_prompt(llm_prompt_name)
        self.current_scene = self._initialize_current_scene()
        self.current_offered_items = []

    def get_current_scene_info(self) -> DialogueScene:
        # 1. アイテム選択
        self.current_offered_items = self._select_items_for_prompt()
        
        # 変数を初期化（NameErrorを防ぐ）
        target_name = "unknown"
        target_item = None

        if self.current_offered_items:
            target_item = self.current_offered_items[0]
            target_name = target_item.name
        else:
            logging.error(f"No items available in Scene ID {self.current_scene.get('id')}")

        # 2. 問いかけの決定
        questions = [
            "What is this color?",
            "How does it taste?"
        ]

        # # --- バイアス回避ロジック: 名前自体に色が含まれる場合、色の質問をスキップ ---
        # if target_item and any(color in target_name.lower() for color in ["green", "white", "orange"]):
        #     color_q = "What color is this？"
        #     if color_q in questions:
        #         questions.remove(color_q)
        #         logging.info(f"Skipped color question for item: {target_name} to avoid bias.")
        # # ----------------------------------------------------------------------

        self.last_question_text = random.choice(questions)
        logging.info(f"Environmental Prompt: {self.last_question_text}")
        
        # 3. 音声生成
        if self.enable_audio_response:
            speakers_description = self.current_scene.get("speaker_description", "")
            response_waveform = self.speech_synthesizer.text_to_waveform(
                text=self.last_question_text, 
                speaker_description=speakers_description
            )
        else:
            response_waveform = np.zeros((self.sampling_rate,), dtype=np.float32)

        # 4. 画像取得
        images, paths = [], []
        for item in self.current_offered_items:
            img, img_path = self._get_item_image(item)
            images.append(img)
            paths.append(img_path)

        self.data_logger.register_scene_paths(paths[:1], self.mode)

        # 5. シーン情報の作成
        dialogue_scene = DialogueScene(
            _scene_id=self.current_scene['id'],
            _prompt_waveform=response_waveform,
            _images=images[:1],
            _prompt_text=self.last_question_text
        )
        # 報酬関数で比較するために「正解名」を保存
        dialogue_scene.target_name = target_name 
        
        return dialogue_scene

    def get_dialogue_feedback(self, agent_speech_waveform: np.ndarray) -> DialogueFeedback:
        # エージェントの発話からDialogueFeedbackを作成
        if not isinstance(agent_speech_waveform, np.ndarray):
            raise ValueError("agent_speech_waveform must be np.ndarray.")

        # 1. 音声認識
        user_input = self.speech_recognizer.speech_to_text(agent_speech_waveform)
        user_input = self.clean_text_for_phonemizer(user_input).strip()
        logging.info(f"Received user input: {user_input}")

        # 2. LLMへの入力（プロンプト）作成
        context_for_llm = f"Your question: '{self.last_question_text}'.\nChild's Answer: '{user_input}'"
        system_prompt = self.generate_system_prompt(dialogue_context=context_for_llm)
        response_text = self.llm.generate_response(system_prompt=system_prompt, user_input="")

        # 3. JSONパース
        json_str = self._extract_json(response_text)
        response_json = {}
        if json_str:
            try:
                response_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error: {e}")

        # LLM抽出情報の取得
        # 卒論のプロンプトはrewardを 1 / -1 で返す。不正な応答は不正解として扱う
        response_quality = 1 if response_json.get("reward") == 1 else 0
        evaluation_logic = response_json.get("evaluation_logic", "")
        item_name_from_llm = response_json.get("item_name", None)
        next_scene_id = response_json.get("next_scene_id", None)

        # 4. 教師側の音声生成
        if self.enable_audio_response:
            response_text_final = response_json.get("response_text", "How may I help you?")
            speakers_description = self.current_scene.get("speaker_description", "")
            response_waveform = self.speech_synthesizer.text_to_waveform(text=response_text_final, speaker_description=speakers_description)
        else:
            response_waveform = np.zeros((self.sampling_rate,), dtype=np.float32)

        self._handle_scene_transition(next_scene_id)

        # 5. ログのアトリビュート表示用アイテム情報
        feedback_item = None
        if self.current_offered_items:
            target_item = self.current_offered_items[0]
            feedback_item = FeedbackItem(
                _id=target_item.id, 
                _image=target_item.image, 
                _attributes=target_item.attributes
            )

        # 6. フィードバック作成と情報の受け渡し
        dialogue_feedback = DialogueFeedback(
            _selected_item=feedback_item,
            _response_waveform=response_waveform
        )
        # 報酬関数が参照する情報を直接属性として持たせる
        dialogue_feedback.llm_item_name = item_name_from_llm
        dialogue_feedback.response_quality = response_quality
        dialogue_feedback.evaluation_logic = evaluation_logic
        dialogue_feedback.response_text = user_input
        
        return dialogue_feedback

    # --- 以下、内部処理用メソッド ---

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        self.scenes = self.config.get('scenes', [])
        self.items = self.config.get('items', [])

    def _load_item_dict_from_config(self) -> Dict[int, Item]:
        item_dict = {}
        for item_data in self.items:
            attributes = item_data.get('attributes', {})
            converted_attributes = {k: (np.float32(v) if isinstance(v, (int, float)) else v) for k, v in attributes.items()}
            item = Item(id=item_data['id'], name=item_data['name'], image=None, prompt=item_data.get('prompt'), attributes=converted_attributes)
            item_dict[item.id] = item
        return item_dict

    def _initialize_current_scene(self) -> Dict[str, Any]:
        initial_scene_id = self.config.get('initial_scene_id', 0)
        return next((s for s in self.scenes if s['id'] == initial_scene_id), None)

    def generate_system_prompt(self, dialogue_context: str = "") -> str:
        if self.current_offered_items:
            target_item = self.current_offered_items[0]
            available_items_text = f"- {target_item.name}: (Attributes: {target_item.attributes})"
        else:
            available_items_text = "No items available."

        scene_dict = {s['id']: s for s in self.scenes}
        transitions = "\n".join(f"- Available scenes: {scene_dict[ns]['name']} (ID: {ns})." for ns in self.current_scene.get("possible_next_scenes", []) if ns in scene_dict)

        return self.prompt_template.format(
            role_description=self.current_scene.get("role_description", "a teacher"),
            scene_information="\n".join(f"- Scene ID: {s['id']}, Scene Name: {s['name']}" for s in self.scenes),
            available_items_text=available_items_text,
            dialogue_context=dialogue_context,
            transitions=transitions,
            guidelines=self.current_scene.get("system_guidelines", "")
        )

    def load_llm_prompt(self, llm_prompt_name):
        template_path = os.path.join(os.path.dirname(__file__), "prompts", llm_prompt_name)
        with open(template_path, encoding="utf-8") as f:
            return f.read()

    def _extract_json(self, text: str) -> Optional[str]:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        return None

    def _handle_scene_transition(self, next_scene_id: Optional[int]):
        if next_scene_id is not None and next_scene_id in self.current_scene.get('possible_next_scenes', []):
            new_scene = next((s for s in self.scenes if s['id'] == next_scene_id), None)
            if new_scene:
                self.current_scene = new_scene

    def _select_items_for_prompt(self) -> List[Item]:
        available_ids = self.current_scene.get('items', [])
        # random.sampleのエラーを防ぐためminを使用
        selected_ids = random.sample(available_ids, min(2, len(available_ids)))
        return [self.data_logger.item_dict[i] for i in selected_ids]

    def _get_item_image(self, item: Item) -> (np.ndarray, str):
        image_index = random.randint(0, self.num_images_per_item - 1)
        image_path = os.path.join(self.data_logger.images_dir, self.mode, f"{item.id}_{item.name}_{image_index}.png")
        if os.path.exists(image_path):
            image = self.data_logger.load_image(item, image_index, self.mode)
        else:
            image = self.image_generator.generate_image(item_name=item.name, item_prompt=item.prompt)
            self.data_logger.save_image(image, item, image_index, self.mode)
        item.image = image
        return image, image_path

    def clean_text_for_phonemizer(self, text):
        return re.sub(f"[^\w\s{re.escape('.,!?' + chr(39) + '_')}]", "", text)