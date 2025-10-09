# MPT-7B-Instruct ファインチューニング プロジェクト

このプロジェクトは、MosaicML社が開発した`mosaicml/mpt-7b-instruct`モデルをベースに、独自のデータセットでファインチューニングを行い、対話形式の推論を実行するためのリポジトリです。

## 特徴

- **データ生成**: Gemini APIを利用して、質問リストから独自のQ&Aデータセットを生成します。
- **ファインチューニング**: PEFT(LoRA)を用いて、効率的にモデルの追加学習を行います。
- **CPU実行対応**: GPUがない環境でも、CPUオフロードを利用してモデルの学習と推論が可能です。（※非常に時間がかかります）

---

## 1. 環境構築

### 1.1. リポジトリのクローン

```bash
git clone https://github.com/hrnk420/AiV2.git
cd AiV2
```

### 1.2. ベースモデルのダウンロード

ファインチューニングの土台となる`mpt-7b-instruct`モデル（約14GB）をダウンロードします。

```bash
# Git LFSがインストールされている必要があります
git clone https://huggingface.co/mosaicml/mpt-7b-instruct ./mpt-7b-instruct
```

### 1.3. 仮想環境の構築とライブラリのインストール

Python 3.9環境での動作を確認しています。

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境のアクティベート (Windowsの場合)
.\venv\Scripts\Activate.ps1

# 必要なライブラリをインストール
pip install -r requirements.txt
```

---

## 2. 使い方

### ステップ1: 学習データの生成 (`datacreate.py`)

1.  `questions.txt`ファイルに、学習させたいQ&Aの「質問」を1行ずつ記述します。
2.  `datacreate.py`を実行する環境で、`GEMINI_API_KEY`という名前の環境変数を設定します。
3.  以下のコマンドを実行すると、Gemini APIが各質問に対する回答を生成し、`data.json`というファイルに保存します。

    ```bash
    python datacreate.py
    ```

### ステップ2: ファインチューニング (`finetuning.py`)

`data.json`を使って、ベースモデルの追加学習を行います。学習が完了すると、モデルの差分データが`lora_output`フォルダに保存されます。

```bash
python finetuning.py
```

**注意:** CPUでの実行には、数時間単位の非常に長い時間がかかります。

### ステップ3: 推論 (`inference.py`)

ファインチューニングしたモデルと対話します。

```bash
python inference.py
```

スクリプトが起動し、「入力してください:」と表示されたら、質問を入力してモデルの応答を確認できます。CPUでの応答生成には数分かかることがあります。

---

##謝辞

このプロジェクトは、以下の素晴らしい技術を利用しています。

- **モデル:** [MosaicML MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)
- **データ生成:** [Google Gemini](https://ai.google.dev/)
- **ライブラリ:** [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [PyTorch](https://pytorch.org/)