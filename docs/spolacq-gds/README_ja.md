# Spolacq-GDS: Spoken Language Acquisition Generative Dialogue Simulator

## 概要
こちらは，[Spolacq-GDS]()の公式実装になります。

## 著者について

[東京科学大学 篠﨑研究室](http://www.ts.ip.titech.ac.jp/)

## 事前準備

Python>=3.10 が必要です。

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

もしvllmのためのwheelのビルドに失敗する場合は，CUDAのバージョンを確認してください。

## 使用方法

仮想環境に入った状態で，次のコマンドを実行してください。

```
python main.py --experiment_config_name yy.yaml
```

この `yy` には，設定ファイル(yaml)の名称を指定してください。
これで無事，学習を始めることができます！
サンプルとして，`food_task.yaml` を指定することができます。

## サンプルコード
当ツールにおいては，対話学習を行うにあたり，
- タスク
- 対話エージェント
の定義が必要になります。
タスクのサンプルコードについては，`tasks` ディレクトリ以下をご覧ください。
対話エージェントのサンプルコードについては，`agents` ディレクトリ以下をご覧ください。

## オリジナルの対話学習を行う
ここでは，オリジナルの対話学習を行うために，必要な操作をご紹介します。

### タスクを定義する
まず，タスクの名称を決め，その名称で `tasks` 以下にディレクトリを作成してください。
その後，作成したディレクトリの中に，次のファイルを作成してください。
- `task_functions.py`
- `xx.json`

`xx` には，タスクの名称を指定してください。

また，次のようなファイルも追加することができます。
- `task_config.json`
- `task_wrappers.py`

#### `task_functions.py`
このファイルには，対話学習を行うにあたって必要な関数を定義します。
具体的には，次のような関数を定義します。

| 関数名 | 返り値 | 役割 |
| - | - | - |
| reward_function | numpy.ndarray | 前の環境と次の環境から，報酬を返します。 |
| termination_function | bool | 前の環境と次の環境から，そのステップでterminationするかを返します。 |
| truncation_function | bool | 次の環境から，そのステップでtruncationするかを返します。 |
| internal_state_update_function | numpy.ndarray | 現時点での内部状態と，対話のフィードバックから，次の内部状態を返します。 |
| observation_function | numpy.ndarray | 次の環境から，観察空間を返します。 |
| initial_internal_state_function | numpy.ndarray | 環境がリセットされるときの内部状態を返します。 |

これらの関数のうちいずれでも定義されていない場合は，エラーが発生します。

#### `xx.json`
このファイルには，タスクの定義を記述します。
トップ階層には，次の値を指定します。
- `initial_scene_id`
- `items`
- `scenes`

`initial_scene_id` には，最初のステップにおけるシーンのidを指定します。シーンについては後述します。

`items` には，この対話で出現するアイテムを，list形式で登録します。
この `items` に登録されるオブジェクトは，次のような値を指定します。

| 名称 | 値 |
| - | - |
| id | アイテムを識別する固有の整数値 |
| name | アイテムの名前 |
| attributes | アイテムに持たせたい性質をオブジェクトで指定 |

`scenes` には，この対話で出現するシーンを，list形式で登録します。
この `scenes` に登録されるオブジェクトには，次のような値を指定します。

| 名称 | 値 |
| - | - |
| id | シーンを識別する固有の整数値 |
| name | シーンの名前 |
| role_description | 対話シミュレータがどのような役割を担うかの説明 |
| speaker_description | 対話シミュレータの返答がどのような話者によるものかの説明 |
| items | このシーンで扱われる可能性のあるアイテムのid |
| possible_next_scenes | このシーンの次のシーンとなりうるシーンのIdを表す。この値と全てのシーンから，オートマトンを構成できる |
| system_guidelines | 対話シミュレータがそのシーンにおいてどのような状況に置かれているかの説明。これがLLMに入力され，対話シミュレータは返答を決定 |

#### `task_config.json`
次のような場合，ディレクトリに `task_config.json` を追加することで対応できます。
- 環境を `gymnasium` の `Wrapper` でラップしたい場合
- 学習のステップ数を指定したい場合

#### `task_wrappers.py`
環境を `gymnasium` の `Wrapper` でラップしたい場合，`task_wrappers.py` を追加することで対応できます。
その場合は，`task_wrappers.py` にラッパーのコードを記述したのち，次のコードを追加してください。

```py
from ..register_functions import make_env_custom_wrap

@make_env_custom_wrap
def wrap_task_env(base_env):
    wrapped_env = YourOriginalWrapper(base_env)
    return wrapped_env
```

### エージェントを定義する
次に，対話学習を行うエージェントを定義する必要があります。
対話学習を行うエージェントを定義するには，まず `agents` 以下にそのエージェント名のディレクトリを作成してください。
次にそのディレクトリの中に
- `agent.py`
- `trainer.py`
- `agent_config.json`

を作成してください。

`agent.py` にはエージェントを，`trainer.py` にはトレーナーを定義します。
その後，`agent_config.json` で，使用するクラスを定義します。
以下は実装例です。

```json
{
    "agent_class": "AgentWithUnits",
    "trainer_class": "Trainer"
}
```
ここでは，エージェントおよびトレーナーのクラス名を明記してください。

`agent.py` および `trainer.py`は，言語や画像などにおいてユニットを使用するか否かで実装法が分岐します。

#### ユニットを使用しないエージェント
単純な，（最も自由度の高い）対話エージェントを学習する場合，`BaseAgent` を継承してエージェントを定義するのが最も簡単です。
この `BaseAgent` は，Spolacq-GDSで対話学習を行うにあたって，必要なAPIを色々整えています。

| 関数名 | 返り値 | 説明 |
| - | - | - |
| pretrain | None | エージェントにおけるモジュールで，事前学習が必要な場合はここに定義 |
| action2speech | np.ndarray | 引数で与えられた強化学習の行動空間(action)を，音声データに変換する関数 |
| get_rl_model | BaseAlgorithm | 強化学習のモデルをここで定義してください。この `BaseAlgorithm` は，`stable_baselines3` のAPIによるものです。
| get_eval_callback | Optional[EvalCallback] | 学習時に与えられる `EvallCallback`。ここでの `EvallCallback` は，`stable_baselines3` のAPIによるものです。 |
| get_progress_callback | Optional[ProgressBarCallback] | 学習時に与えられる `ProgressCallback`。ここでの `ProgressCallback` は，Spolacq-GDSが独自に実装したCallbackです。 |

その後，`BaseTrainer` を継承してトレーナーを定義するのが簡便です。
`BaseTrainer` は次のようなAPIを持ちます。

| 関数名 | 返り値 | 説明 |
| - | - | - |
| pretrain | None | エージェントにおけるモジュールで，事前学習が必要な場合はここに定義 |
| train_rl | None | エージェントを強化学習する際の手順を定義する関数 |

実装例は，`agents` 以下の `sample01` をご覧ください。
こちらは，[Continuous Action Space-Based Spoken Language Acquisition Agent Using Residual Sentence Embedding and Transformer Decoder](https://ieeexplore.ieee.org/document/10096250)のエージェントを実装したものです。

#### ユニットを使用するエージェント
定義するファイルはユニットを使用しないバージョンと同じですが，ユニットを使用する場合は 
`BaseAgentWithUnits` クラスを継承して，エージェントを実装できます。そうすると，上述の `BaseAgent` のAPIに加えて，次のようなプロパティも提供され，より便利になります。

| 名前 | 型/返り値 | 説明 |
| - | - | - |
| i2u | BaseImage2Unit | Image2Unit のインスタンス |
| s2u | BaseSpeech2Unit | Speech2Unit のインスタンス |
| u2s | BaseUnit2Speech | Unit2Speech のインスタンス |
| train_i2u | None | I2Uモデルを訓練する場合定義してください |
| train_u2s | None | U2Sモデルを訓練する場合定義してください |
| train_s2u | None | S2Uモデルを訓練する場合定義してください |

### 学習コードを実行する
これが最後のステップです！
学習を実行するために，設定ファイルを `experiments` ディレクトリ内に作成してください。
このファイルの名称は任意でよいですが，yamlファイルにしてください。

このyamlファイルには，次の値を指定してください。

| 名称 | 値 |
| - | - |
| task_name | どのタスクを利用するか。`tasks` ディレクトリ内のディレクトリ名を指定してください |
| agent_name | どのエージェントを利用するか。`agents` ディレクトリ内のディレクトリ名を指定してください |
| device | `cpu` または `cuda` を指定。特に理由が無ければ `cuda` を推奨 |
| data_dir_id (任意) | そのディレクトリ(`data/<task_name>/`)以下にデータが保存される |

また，`Agent` を指定してください。
この `Agent` の階層には，エージェントに与える設定を指定することができます。

最後に，次のようにコードを実行してください。

```
python main.py --experiment_config_name yy
```

この `yy` には，設定ファイル(yaml)の名称を指定してください。
これで無事，学習を始めることができます！

## ライセンス
Copyright (C) 2025- Shinozaki Lab Science Tokyo

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## 公開予定
- APIリファレンス
