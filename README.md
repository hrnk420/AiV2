# AI-v2: Phi-2 & Gemini 連携ファインチューニング・システム

このプロジェクトは、Googleの **Gemini 1.5 Flash API** を使用して高品質な学習データを自動生成し、Microsoftの軽量・高性能モデル **Phi-2** をベースに **LoRA (Low-Rank Adaptation)** を用いて独自の微調整（ファインチューニング）を行うための統合ツールキットです。

Windows環境のCPUのみでも動作するように最適化されており、GPU（CUDA）があれば自動的に検知して高速な学習・量子化推論を行います。

---

## 主な特徴

- **自動データ生成 (`datacreate.py`)**: Gemini 1.5 Flash API を活用し、質問リストからプログラミング解説データセットを自動構築。
- **軽量・高性能モデル (`Phi-2`)**: 27億パラメータながら高い性能を持つ `microsoft/phi-2` を採用。低メモリ環境でも動作可能。
- **ハードウェア自動最適化**: CUDAが利用可能な場合は、4ビット量子化（bitsandbytes）とFP16演算を自動適用。
- **一気通貫のワークフロー**: データの生成、LoRAによる学習、学習済みモデルとの対話までをシンプルなスクリプトで完結。

---

## 1. セットアップ

### 1.1. リポジトリの準備
```bash
git clone https://github.com/hrnk420/AiV2.git
cd AiV2
```

### 1.2. 環境構築
Python 3.10〜3.12 環境を推奨します。
```bash
# 仮想環境の作成
python3.12 -m venv venv  3.12の場合
# アクティベート (Windows)
.\venv\Scripts\activate

# 必要なライブラリのインストール
pip install -r requirements.txt
```

### 1.3. APIキーの設定
プロジェクトのルートディレクトリに `.env` ファイルを作成し、Gemini APIキーを記述してください。
```text
GEMINI_API_KEY=your_google_api_key_here
```

---

## 2. 使い方

### ステップ1: 学習データの生成 (`datacreate.py`)
`questions.txt` に記述された質問に基づき、Gemini APIが回答を生成し `data.json` に保存します。
```bash
python datacreate.py
```
※ 無料枠のAPI制限（RPM）を考慮し、安全な待機時間を設けています。

### ステップ2: ファインチューニング (`finetuning.py`)
生成された `data.json` を使用して、Phi-2モデルにLoRA微調整を施します。
```bash
python finetuning.py
```
- **GPU環境**: 自動的に4ビット量子化が行われ、高速に学習が進みます。
- **CPU環境**: オフロード機能を利用して学習を行います（時間がかかる場合があります）。
- 学習済みパラメータは `./lora_output_phi2` に保存されます。

### ステップ3: AIとの対話 (`inference.py`)
学習したモデルを読み込んで、対話形式で動作を確認します。
```bash
python inference.py
```

---

## 3. 推奨設定と注意点

- **データ量**: 高品質な回答を得るためには、最低でも 100〜200 件程度のデータを `data.json` に蓄積することを推奨します。
- **制限事項**: Gemini APIの無料枠には1日のリクエスト上限があります。制限に達した場合は翌日に再開してください。
- **セキュリティ**: `.env` や学習済みの巨大なモデルファイルは `.gitignore` により保護されており、GitHubにはアップロードされません。

---

## 謝辞と技術スタック

- **Base Model:** [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
- **Data Generation:** [Google Gemini API](https://ai.google.dev/)
- **Libraries:** [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [PyTorch](https://pytorch.org/)

---
Created by [hrnk420](https://github.com/hrnk420)
