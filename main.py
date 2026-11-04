
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import numpy as np
import torch
from pathlib import Path
import yaml
import gymnasium as gym


from env.create_image_dataset import create_image_dataset
from env.create_audio_dataset import create_audio_dataset
# from agents.sample01.tools.I2U.create_input_files import create_input

from env.utils import load_common_components
from tasks.get_env import get_env, get_task_path
from agents.get_agent import get_agent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(experiment_config_name):
    # 実験設定の読み込み
    experiment_config_path = Path('experiments', experiment_config_name)
    print('experiment_config_path:', experiment_config_path)
    experiment_config = yaml.safe_load(experiment_config_path.read_text())
    print('experiment_config:', experiment_config)
    return experiment_config

def load_task_info(base_config):
    # itemsのconfigファイルのパスを取得
    task_name = base_config['task_name']
    task_path = get_task_path(task_name)
    task_setting_path = task_path["config_file"]

    # task setting fileの読み込み
    with open(task_setting_path, 'r', encoding='utf-8') as file:
        task_setting = yaml.safe_load(file)
    # itemsの取得
    items = [item['name'] for item in task_setting['items']]
    # itemsのpromptを取得
    item_prompts = [item['prompt'] if 'prompt' in item else None for item in task_setting['items']]
    return task_name, task_path, task_setting, items, item_prompts


def set_base_dir(base_config):
    if 'task_name' not in base_config:
        raise ValueError("task_name is not found in experiment config.")
    if 'agent_name' not in base_config:
        raise ValueError("agent_name is not found in experiment config.")
    
    base_data_dir = Path('data', base_config['task_name'])
    base_agent_dir = Path('agents', base_config['agent_name'])
    # base_data_dir以下に、00000や00001などのディレクトリをつくる。
    # data_dir_idが指定されている場合は、そのディレクトリを使う。
    if 'dir_id' in base_config:
        base_data_dir = base_data_dir / base_config['dir_id']
        base_agent_experiment_dir = base_agent_dir / base_config['task_name'] / base_config['dir_id']
    else:
        dir_id = max(len(list(base_data_dir.iterdir())), len(list((base_agent_dir / base_config['task_name']).iterdir())))
        base_data_dir = base_data_dir / f"{dir_id:05d}"
        base_agent_experiment_dir = base_agent_dir / base_config['task_name'] / f"{dir_id:05d}"
    base_data_dir.mkdir(parents=True, exist_ok=True)
    base_agent_dir.mkdir(parents=True, exist_ok=True)
    base_agent_experiment_dir.mkdir(parents=True, exist_ok=True)
    return base_data_dir, base_agent_dir, base_agent_experiment_dir

    

def main():
    parser = argparse.ArgumentParser(description='Set seeds for reproducibility in PyTorch.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default: 1234)')
    parser.add_argument('--experiment_config_name', type=str, default='food_task.yaml', help='Experiment config file name (default: food_task.yaml)')
    args = parser.parse_args()

    # シードを設定
    seed = args.seed
    print(f"Setting random seed to {seed}")
    set_seed(seed)

    # 実験設定の読み込み
    experiment_config = load_config(args.experiment_config_name)

    # タスク設定の読み込み
    base_config = experiment_config['base']
    task_name, task_path, task_setting, items, item_prompts = load_task_info(base_config)

    # データ保存用のディレクトリを作成
    data_dir, agent_dir, agent_experiment_dir = set_base_dir(base_config)

    # 事前学習
    pretrain_data_dir = data_dir / 'pretrain'
    os.makedirs(pretrain_data_dir, exist_ok=True)
    pretrain_image_config = experiment_config['pretrain']['image']


    # 画像データセットの作成
    image_data_dir = pretrain_data_dir / 'Image'
    image_paths = create_image_dataset(
        image_data_dir=image_data_dir,
        model_name=pretrain_image_config['model_name'],
        model_config=pretrain_image_config['model'],
        items=items,
        item_prompts=item_prompts,
        train_images_per_folder=pretrain_image_config['train_images_per_folder'],
        test_images_per_folder=pretrain_image_config['test_images_per_folder'],
        overwrite=pretrain_image_config['overwrite_image_dataset'],
        batch_size=pretrain_image_config['batch_size'],
    )
    

    # 音声データセットの作成
    audio_data_dir = pretrain_data_dir / 'Audio'
    audio_config = experiment_config['pretrain']['audio']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_paths = create_audio_dataset(
        task_name,
        image_data_dir,
        audio_data_dir,
        audio_config,
        device,
        train_images_per_folder=pretrain_image_config['train_images_per_folder'],
        test_images_per_folder=pretrain_image_config['test_images_per_folder'],
    )


    # pretrain
    pretrain_config = experiment_config['pretrain']
    agent_name = base_config['agent_name']
    agent, trainer = get_agent(agent_name, pretrain_config)
    
    pretrain_info = trainer.pretrain(
        agent,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir=agent_dir,
        agent_experiment_dir=agent_experiment_dir,
    )



    # RL
    rl_data_dir = data_dir / 'rl'
    rl_config = experiment_config['rl']

    rl_env_config = rl_config['env']

    #pretrainと同じサンプリングレートを設定
    sampling_rate = pretrain_config['U2S']['u2m']['audio']['sampling_rate']

    # 共通のコンポーネントの読み込み
    common_models, data_logger = load_common_components(
        base_data_dir=rl_data_dir,
        sampling_rate=sampling_rate,
        seed=seed,
        device=rl_env_config['device'],
        tts_model=rl_env_config.get('tts_model', 'parler'),
    )
    llm, speech_recognizer, speech_synthesizer, image_generator = common_models

    # 環境の初期化
    env_train = get_env(
        base_config['task_name'],
        data_logger,
        llm,
        speech_recognizer,
        speech_synthesizer,
        image_generator,
        llm_prompt_name=rl_env_config['llm_prompt_name'],
        mode="train",
        record_mode=rl_env_config['record_mode'],
        num_images_per_item=rl_env_config['num_images_per_item'],
        sampling_rate=sampling_rate,
        enable_audio_response=rl_env_config['enable_audio_response'],
    )

    env_test = get_env(
        base_config['task_name'],
        data_logger,
        llm,
        speech_recognizer,
        speech_synthesizer,
        image_generator,
        llm_prompt_name=rl_env_config['llm_prompt_name'],
        mode="test",
        record_mode=rl_env_config['record_mode'],
        num_images_per_item=rl_env_config['num_images_per_item'],
        sampling_rate=sampling_rate,
        enable_audio_response=rl_env_config['enable_audio_response']
    )


    agent_config = rl_config['agent']
    env_config = rl_config['env']

    trainer.train_rl(
        agent,
        env_train,
        env_test,
        base_config,
        env_config,
        agent_config,
        pretrain_info,
        agent_dir=agent_dir,
        agent_experiment_dir=agent_experiment_dir,
    )




if __name__ == "__main__":
    main()
