# Spolacq-GDS: Spoken Language Acquisition Generative Dialogue Simulator

## Overview
This is the official implementation of [Spolacq-GDS]().

## About auther

[Shinozaki Lab Science Tokyo](http://www.ts.ip.titech.ac.jp/)

## Preparation

Need Python>=3.10.

Linux/macOS
```
python -m venv spolacq_gds_env
source spolacq_gds_env/bin/activate
pip install -r requirements.txt
```

Windows
```
python -m venv spolacq_gds_env
./spolacq_gds_env/Scripts/activate
pip install -r requirements.txt
```

If you fail with building wheel for vllm, please check your CUDA version.

## Usage
Please run the following command in your virtual environment.

```
python main.py --experiment_config_name yy.yaml
```

On `yy`, please specify the name of the config file(yaml).
Then, you could start the learning!
You can specify `food_task.yaml` as sample.

## Sample codes
On this tool, to have dialogue learning, you have to define
- Task
- Dialogue agent

You can see sample codes of task by visiting under `tasks` directory.
You can see sample codes of dialogue agent by visiting under `agents` directory.

## Have original dialogue learning
Here we will introduce needed operations to have original dialogue learning.

### Define tasks
At first, please decide the name of the task, and make the directory whose name is that.
After that, please make following files in it.
- `task_functions.py`
- `xx.json`

On `xx`, please specify the name of the task.

Also, you can add following files.
- `task_config.json`
- `task_wrappers.py`

#### `task_functions.py`
On this file, please define the function needed to have dialogue learning.
To be detailed, following functions are defined.

| Function | Return value | Role |
| - | - | - |
| reward_function | numpy.ndarray | Returns reward according to the previous environment and the next one. |
| termination_function | bool | Returns whether it terminates according to the previous environment and the next one. |
| truncation_function | bool | Returns whether it truncates according to the next environment. |
| internal_state_update_function | numpy.ndarray | Returns next internal state according to the internal state on this step and the feedback of dialogue. |
| observation_function | numpy.ndarray | Returns observation space according to the next environment. |
| initial_internal_state_function | numpy.ndarray | Returns the internal state when the environment resets. |

The error will occur if any function on the above isn't defined.

#### `xx.json`
On this file please define the task.
On the top directory, please specify the following values.
- `initial_scene_id`
- `items`
- `scenes`

On `initial_scene_id`, please specify the id of the scene of the first step. We will explain scene later.

On `items`, please register the items that will appear on this dialogue, with list format.
You will specify the following values in `items`.

| name | value |
| - | - |
| id | Integer that identifies the item |
| name | Name of the item |
| attributes | Attributes you want to have the item to take |

On `scenes`, please register the scene that will appear on this dialogue, with list format.
You will specify the following values in `scenes`.

| name | value |
| - | - |
| id | Integer that identifies the scene |
| name | Name of the scene |
| role_description | What role this dialogue simulator plays |
| speaker_description | What characteristics of person replies by this dialogue simulator |
| items | Ids of items that might be treated on this scene |
| possible_next_scenes | Ids that will be the next scene of this. From this value and the all of scenes, you can construct automaton |
| system_guidelines | The prompt that explains what situation this dialogue simulator is on. This will be input to LLM and the dialogue simulator decides what to reply |

#### `task_config.json`
You can add `task_config.json` to cope with the following situations.
- Wrap with the environment by `Wrapper` of `gymnasium`.
- Specify the number of steps of learning

#### `task_wrappers.py`
You can add `task_wrappers.py` to wrap with the environment by `Wrapper` of `gymnasium`.
If so, please add following codes after the definition of the wrapper, on `task_wrappers.py`.

```py
from ..register_functions import make_env_custom_wrap

@make_env_custom_wrap
def wrap_task_env(base_env):
    wrapped_env = YourOriginalWrapper(base_env)
    return wrapped_env
```

### Define agent
Next, you have to define the agent that will learn dialogue.
To define dialogue learning agent, please make the directory whose name is the name of the agent under `agents`.
Next please make the following file in it.
- `agent.py`
- `trainer.py`
- `agent_config.json`

You have to define agent on `agent.py` and trainer on `trainer.py`.
Then you define used classes on `agent_config.json`.
This is the example.

```json
{
    "agent_class": "AgentWithUnits",
    "trainer_class": "Trainer"
}
```

Here you should define the name of the classes of agent and trainer.

From here, you diverge implementing ways whether you use units on language or image and so on.

#### Agents without using units
To learn the simplest (and the most free) dialogue agent, it's easiest to use the agent that inherits `BaseAgent`.
This `BaseAgent` has some needed APIs to have dialogue learning.

| name | return value | explanation |
| - | - | - |
| pretrain | None | If you have some modules that needs to be pretrained, please define this function |
| action2speech | np.ndarray | It converts action space of reinforcement learning to audio data |
| get_eval_callback | Optional[EvalCallback] | `EvalCallback` passed to train. This `EvallCallback` is one of the APIs of `stable_baselines3` |
| get_progress_callback | Optional[ProgressCallback] | `ProgressCallback` passed to train. This `ProgressCallback` is originally defined on Spolacq-GDS |

Then, it's simple to define trainer, with inheriting `BaseTrainer`.
`BaseTrainer` has following APIs.

| name | return value | explanation |
| - | - | - |
| pretrain | None | If you have some modules that needs to be pretrained, please define this function |
| train_rl | None | Function that has procedure of reinforcement learning of agent |

Please refer to `sample01` under `agents` as an example of implementation. This is the agent proposed on [Continuous Action Space-Based Spoken Language Acquisition Agent Using Residual Sentence Embedding and Transformer Decoder](https://ieeexplore.ieee.org/document/10096250).

#### Agents with using units

The files to be defined are same, but if you use units, you can define agent with inheriting `BaseAgentWithUnits`. Then you can define easier because you can use the following properties.

| name | type/return value | explanation |
| - | - | - |
| i2u | BaseImage2Unit | The instance of Image2Unit |
| s2u | BaseSpeech2Unit | The instance of Speech2Unit |
| u2s | BaseUnit2Speech | The instance of Unit2Speech |
| train_i2u | None | Please define if you train I2U model |
| train_u2s | None | Please define if you train U2S model |
| train_s2u | None | Please define if you train S2U model |

### Run the learning code
This is the final step!
To learn, please make config file in `experiments` directory.
The name can be arbitrary, but it should be yaml file.

On this yaml file, please specify the following values.

| name | value |
| - | - |
| task_name | Which task you will use. It should be the name of the directory in `tasks` directory |
| agent_name | Which agent you will use. It should be the name of the directory in `agents` directory |
| device | Please specify `cpu` or `cuda`. Unless there is a specific reason, we recommend to specify `cuda` |
| data_dir_id (optional) | The data will be stored under the specified directory (`data/<task_name>/`) |

Also, please specify `Agent`.
On the `Agent` layer, you can specify the setting that you want to give to the agent.

After that, please run the code as follows:

```
python main.py --experiment_config_name yy.yaml
```

On `yy`, please specify the name of the config file(yaml).
Then, you could start the learning!

## License
Copyright (C) 2025- Shinozaki Lab Science Tokyo

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Coming soon
- API Reference