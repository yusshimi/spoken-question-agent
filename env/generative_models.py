import os
from typing import Any
import numpy as np
import librosa
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, VitsModel, pipeline
from parler_tts import ParlerTTSForConditionalGeneration
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
from vllm import LLM, SamplingParams

class LLMComponent:
    def __init__(self, seed: int | None = None, device: str = "cuda"):
        if seed is None:
            seed = 1234
        torch.random.manual_seed(seed)  # シードの設定

        # モデルの読み込み
        # Qwen2.5-7B-Instruct に変更
        self.llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct", # ここを変更
            max_model_len=2048,               # Qwenはコンテキストが長いため、少し余裕を持たせてもOK
            gpu_memory_utilization=0.5,       # 7Bモデルにする場合、0.35だと足りない可能性があるので少し増やします
            download_dir="./cache/LLM",
            trust_remote_code=True            # Qwen等のモデルで必要な場合があります
        )
        self.tokenizer = self.llm.get_tokenizer()
    
    def generate_response(self, system_prompt: str, user_input: str) -> dict[str, Any]:
        # LLMへの入力の設定
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,                # トークナイズせず文字列として生成
            add_generation_prompt=True     # 生成用プロンプトを追加
            )
        #もともと150
        generation_args = {
            "max_new_tokens": 200,
            "temperature": 0.5,
            }
        
        sampling_params = SamplingParams(
            max_tokens=generation_args["max_new_tokens"],
            temperature=generation_args["temperature"]
            )
        
        output = self.llm.generate(prompt, sampling_params)

        return output[0].outputs[0].text

class SpeechRecognizerComponent:
    def __init__(self, sampling_rate: int = 22050, device: str = "cuda"):
        # ASRモデルの読み込み
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir="./cache/ASR",
        ).to(device)
        self.asr_processor = AutoProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir="./cache/ASR",
        )
        self.asr_model.config.forced_decoder_ids = None
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            device=device,
        )
        self.sampling_rate = sampling_rate
    
    def speech_to_text(self, waveform: np.ndarray) -> str:
        # 音声をテキストに変換
        recognized_text = self.asr_pipe({"sampling_rate": self.sampling_rate, "raw": waveform}, generate_kwargs={"language": "english"})["text"]

        return recognized_text

class SpeechSynthesizerComponent:
    def __init__(self, sampling_rate: int = 22050, device: str = "cuda"):
        # TTSモデルの読み込み
        self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1",
            cache_dir="./cache/TTS",
        ).to(device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained(
            "parler-tts/parler-tts-mini-v1",
            cache_dir="./cache/TTS",
        )
        default_sampling_rate = self.tts_model.config.sampling_rate
        if sampling_rate != default_sampling_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=default_sampling_rate,
                new_freq=sampling_rate
            ).to(device)
        else:
            self.resampler = None
    
    def text_to_waveform(self, text: str, speaker_description: str, device: str = "cuda") -> np.ndarray:
        # モデルへの入力の設定
        input_ids = self.tts_tokenizer(speaker_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.tts_tokenizer(text, return_tensors="pt").input_ids.to(device)

        # 音声の生成
        generation = self.tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        waveform_tensor = generation.squeeze(0)

        if self.resampler is not None:
            waveform_tensor = self.resampler(waveform_tensor.unsqueeze(0)).squeeze(0)

        return waveform_tensor.cpu().numpy()

class VitsSpeechSynthesizerComponent:
    """VITS-LJSによる音声合成。SpeechSynthesizerComponentと同じインターフェースを持つ"""

    def __init__(self, sampling_rate: int = 22050, device: str = "cuda"):
        self.device = device
        self.tts_model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs", cache_dir="./cache/TTS").to(device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs", cache_dir="./cache/TTS")
        default_sampling_rate = self.tts_model.config.sampling_rate
        if sampling_rate != default_sampling_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=default_sampling_rate,
                new_freq=sampling_rate
            ).to(device)
        else:
            self.resampler = None

    def text_to_waveform(self, text: str, speaker_description: str = "", device: str | None = None) -> np.ndarray:
        # VITS-LJSは単一話者のためspeaker_descriptionは使わない
        inputs = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            waveform_tensor = self.tts_model(**inputs).waveform

        if self.resampler is not None:
            waveform_tensor = self.resampler(waveform_tensor)

        return waveform_tensor.squeeze(0).cpu().numpy()

class ImageGeneratorComponent:
    def __init__(self, device: str = "cuda"):
        # モデルの読み込み
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            cache_dir="./cache/image_generator",
        ).to(device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.load_lora_weights(
            "mann-e/Mann-E_Turbo",
            weight_name="manne_turbo.safetensors",
            cache_dir="./cache/image_generator",
        )
        self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )
    
    def generate_image(self, item_name: str, item_prompt: str | None = None) -> np.ndarray:
        # プロンプトの設定
        if item_prompt is None:
            prompt = f"A {item_name} on a white background, uncooked, realistic."
        else:
            prompt = item_prompt
        
        num_inference_steps = 8  # 推論ステップの設定
        guidance_scale = 4  # ガイダンススケールの設定
        image_height = 512  # 画像の高さ
        image_width = 512  # 画像の幅
        image_output_type = "np"  #画像の出力形式の設定

        # 画像の生成
        img_arr_float32 = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=image_height,
            width=image_width,
            output_type=image_output_type,
        ).images[0]
        img_arr_uint8 = (img_arr_float32 * 255).clip(0, 255).astype("uint8")

        return img_arr_uint8


def load_generative_models(sampling_rate: int = 22050, seed: int | None = None, device: str = "cuda", tts_model: str = "parler") -> tuple[LLMComponent, SpeechRecognizerComponent, SpeechSynthesizerComponent | VitsSpeechSynthesizerComponent, ImageGeneratorComponent]:
    speech_recognizer = SpeechRecognizerComponent(sampling_rate, device=device)
    if tts_model == "vits-ljs":
        speech_synthesizer = VitsSpeechSynthesizerComponent(sampling_rate, device=device)
    elif tts_model == "parler":
        speech_synthesizer = SpeechSynthesizerComponent(sampling_rate, device=device)
    else:
        raise ValueError(f"Unknown tts_model: {tts_model}")
    image_generator = ImageGeneratorComponent(device=device)
    llm = LLMComponent(seed=seed, device=device)

    return llm, speech_recognizer, speech_synthesizer, image_generator
