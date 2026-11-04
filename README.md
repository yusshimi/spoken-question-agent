# 音声質問を理解する自律学習エージェントの検討

学部の卒業研究（学士特定課題研究）で扱ったコードです。

## このコードがしていること

**音声で聞かれた質問に対する適切な回答を、正解データなしでAIエージェントに学ばせる実験**のコードです。

赤ちゃんが親とのやり取りから言葉を覚える過程を、計算機上で再現する研究分野（音声言語獲得）に属します。エージェントには文字のラベルを与えず、音声と画像、そして「答えが合っていたか」という報酬だけを手がかりに学習させます。

たとえば、りんごの画像を見せながら音声で質問します。

| 質問（音声） | 期待する答え（音声） |
|---|---|
| What is this color? | This apple looks red. |
| How does it taste? | This apple is very sweet. |

見せる画像は同じなので、答えを決めるには質問を聞き分ける必要があります。この研究では、エージェントが報酬だけを頼りに、質問に応じて答えを変えられるようになるかを検証しました。

## 背景

先行研究には、次の課題がありました。

- 音声質問を理解できるエージェントは、「欲しいもの」と「欲しくないもの」を問う単純な質問に限られていた
- 複数の単語を話せる新しいエージェント（小松ら, ICASSP 2023）は、質問を聞き取る機能を持っていなかった

そこで、後者のエージェントに音声の入力を統合し、質問を理解しないと報酬を得られないタスクを新たに設計しました。

## 仕組み

学習は2段階です。

1. **事前学習:** 画像と回答音声のペアから、発話の生成を学びます。
2. **対話学習:** 環境と対話し、報酬から質問に応じた発話の選び方を学びます（強化学習）。

対話学習の流れは次のとおりです。

```
質問の音声 → HuBERT（音声の特徴を抽出）→ Actor（何を言うかを決める）─┐
                                                                    ├→ 発話を生成 → 音声認識
りんごの画像 → DINO（画像の特徴を抽出）─────────────────────────────┘        ↓
                                                                       LLMが回答を判定
Actor / Criticを更新 ←──────── 報酬（適切なら+1、不適切なら-1）←──────────────┘
```

| 要素 | 使用した技術 |
|---|---|
| 質問音声の特徴抽出 | HuBERT（自己教師あり学習済みモデル） |
| 画像の特徴抽出 | DINO（ViT-B/8） |
| 強化学習 | DDPG（Actor-Critic）、Stable-Baselines3をベースに実装 |
| 発話の生成 | Transformer Decoder、Tacotron2、HiFi-GAN |
| 環境（対話シミュレータ） | Spolacq-GDS、Qwen2.5-7B（判定）、Whisper（音声認識）、VITS（質問音声） |
| 言語、フレームワーク | Python、PyTorch、Gymnasium |

## 結果

卒論の実験では、**エージェントはタスクを達成できませんでした。**

- 平均報酬は学習全体を通じて0付近にとどまりました（理想は1.0）。
- 学習初期には複数の回答を発話していましたが、後半には質問に関係なく、特定の回答1つに固定されました。
- 複数の回答を発話できていたことから、発話の生成はある程度できており、質問に応じた切り替えができていないと考えました。

卒論では、学習済みの音声特徴を接続するだけで、報酬のみから質問の識別能力を獲得させることは難しいと結論づけました。改善案として、音声特徴の事前学習と、確率的な方策を使うSACの導入を挙げています。

## 自分が担当した部分

ベースのコードに対して、次の部分を設計、実装しました。

| 内容 | 主なファイル |
|---|---|
| タスクの設計（対話の定義、回答パターン、報酬） | [`tasks/food_task/`](tasks/food_task/) |
| LLMによる回答判定のプロンプト | [`env/prompts/food_task.txt`](env/prompts/food_task.txt)、[`env/dialogue_simulator.py`](env/dialogue_simulator.py) |
| HuBERTの統合と特徴抽出 | [`tasks/food_task/task_wrappers.py`](tasks/food_task/task_wrappers.py)、[`agents/sample01/utils/sb3_api.py`](agents/sample01/utils/sb3_api.py) |
| 状態と行動の変更、探索ノイズの減衰 | [`agents/sample01/agent.py`](agents/sample01/agent.py)、[`agents/sample01/trainer.py`](agents/sample01/trainer.py)、[`agents/sample01/utils/callbacks.py`](agents/sample01/utils/callbacks.py) |
| 実験設定と集計 | [`experiments/food_task.yaml`](experiments/food_task.yaml)、[`scripts/summarize_rewards.py`](scripts/summarize_rewards.py) |

## コードの読み方

| 場所 | 内容 |
|---|---|
| [`main.py`](main.py) | 入口。データ生成、事前学習、対話学習を順に実行 |
| [`experiments/`](experiments/) | 実験の設定 |
| [`tasks/food_task/`](tasks/food_task/) | タスクの定義、報酬、音声の前処理 |
| [`env/`](env/) | 対話シミュレータ、LLMや音声認識などの生成モデル |
| [`agents/sample01/`](agents/sample01/) | エージェント本体と学習処理 |
| [`agents/sample01/tools/`](agents/sample01/tools/) | 発話生成や画像特徴抽出で使う外部コード |

## 実行方法

GPUでの全工程の実行が必要です。このリポジトリの状態では、手順を通しで実行した確認はしていません。

**必要な環境**

- Linux、NVIDIA GPU（CUDA）
- Python 3.10、[Poetry](https://python-poetry.org/)
- `ffmpeg`、`espeak-ng`

**準備**

```bash
poetry install

# HiFi-GANの重みを取得する（Spolacq-GDSで公開されているものを使用）
curl -L -o agents/sample01/models/U2S/m2s/g_02500000 \
  https://raw.githubusercontent.com/tttslab/spolacq/spolacq-GDS/agents/sample01/models/U2S/m2s/g_02500000
```

**実行**

```bash
poetry run python main.py --experiment_config_name food_task.yaml
```

## ベースにしたコードとライセンス

東京科学大学 篠﨑研究室の [Spolacq-GDS](https://github.com/tttslab/spolacq) を改変したもので、Apache License 2.0 で公開しています。詳細は [LICENSE](LICENSE) と [NOTICE](NOTICE) をご覧ください。Spolacq-GDSの元の説明は [docs/spolacq-gds/](docs/spolacq-gds/) にあります。

`agents/sample01/tools/` 以下などに含まれる外部コードには、それぞれのライセンスが適用されます。

## 参考文献

- R. Komatsu, Y. Kimura, T. Okamoto, T. Shinozaki, "Continuous Action Space-Based Spoken Language Acquisition Agent Using Residual Sentence Embedding and Transformer Decoder," ICASSP 2023.
- K. Toyoda et al., "Self-Supervised Spoken Question Understanding and Speaking with Automatic Vocabulary Learning," O-COCOSDA 2021.
- W.-N. Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," IEEE/ACM TASLP, 2021.
