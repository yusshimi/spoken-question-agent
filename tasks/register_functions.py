import gymnasium as gym
from functools import wraps

from env.dialogue_environment_state_manager import AgentStateManager
from env.dialogue_simulator import DialogueSimulator
from .wrappers import wrap_custom_functions, wrap_logging



def make_env_base(
        data_logger, 
        llm, 
        speech_recognizer,
        speech_synthesizer,
        image_generator,
        config_path,
        llm_prompt_name,
        mode,
        num_images_per_item,
        sampling_rate,
        enable_audio_response,
        initial_internal_state_function,
        max_step,
        reward_function,
        termination_function,
        truncation_function,
        internal_state_update_function,
        observation_function,
        record_mode=False
    ):
    print('make_env_base')
    print('config_path:', config_path)
    dialogue_simulator = DialogueSimulator(
        config_path=config_path, 
        llm=llm, 
        speech_recognizer=speech_recognizer,
        speech_synthesizer=speech_synthesizer,
        image_generator=image_generator, 
        data_logger=data_logger,
        llm_prompt_name=llm_prompt_name,
        mode=mode,
        num_images_per_item=num_images_per_item,
        sampling_rate=sampling_rate,
        enable_audio_response=enable_audio_response
    )

    asm = AgentStateManager(
        dialogue_simulator=dialogue_simulator,
        initial_internal_state_function=initial_internal_state_function,
        max_step=max_step
    )

    base_env = wrap_custom_functions(
        asm,
        reward_function,
        termination_function=termination_function,
        truncation_function=truncation_function,
        internal_state_update_function=internal_state_update_function,
        observation_function=observation_function,
    )

    base_env = wrap_logging(
        base_env, 
        record_mode=record_mode, 
        data_logger=data_logger,
        mode=mode,
    )


    return base_env


def make_env_custom_wrap(make_wrapper_fn):
    """
    基本環境ファクトリー関数とラッパーを自動的に組み合わせるデコレータ
    
    使用例:
    @make_env_custom_wrap
    def make_task_env(base_env, custom_param=None):
        return TaskWrapper(base_env, custom_param)
    """
    @wraps(make_wrapper_fn)
    def wrapper(*args, **kwargs):
        # 基本環境パラメータを取り出す
        data_logger = kwargs.pop('data_logger', None)
        llm = kwargs.pop('llm', None)
        speech_recognizer = kwargs.pop('speech_recognizer', None)
        speech_synthesizer = kwargs.pop('speech_synthesizer', None)
        image_generator = kwargs.pop('image_generator', None)

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

        config_path = kwargs.pop('config_path', None)
        mode = kwargs.pop('mode', None)
        num_images_per_item = kwargs.pop('num_images_per_item', None)
        sampling_rate = kwargs.pop('sampling_rate', None)
        initial_internal_state_function = kwargs.pop('initial_internal_state_function', None)
        max_step = kwargs.pop('max_step', None)


        reward_function = kwargs.pop('reward_function', None)
        termination_function = kwargs.pop('termination_function', None)
        truncation_function = kwargs.pop('truncation_function', None)
        internal_state_update_function = kwargs.pop('internal_state_update_function', None)
        observation_function = kwargs.pop('observation_function', None)


        record_mode = kwargs.pop('record_mode', False)


        
        # 基本環境を作成
        base_env = make_env_base(
            data_logger=data_logger, 
            llm=llm, 
            speech_recognizer=speech_recognizer,
            speech_synthesizer=speech_synthesizer,
            image_generator=image_generator,
            config_path=config_path,
            mode=mode,
            num_images_per_item=num_images_per_item,
            sampling_rate=sampling_rate,
            initial_internal_state_function=initial_internal_state_function,
            max_step=max_step,
            reward_function=reward_function,
            termination_function=termination_function,
            truncation_function=truncation_function,
            internal_state_update_function=internal_state_update_function,
            observation_function=observation_function,
            record_mode=record_mode
        )

        # 残りの引数でラッパーを適用
        return make_wrapper_fn(base_env, **kwargs)
    
    return wrapper





@make_env_custom_wrap
def wrap_task_env(base_env, custom_param=None):
    return base_env


