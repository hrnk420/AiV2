import google.generativeai as genai
import os
import json
import time
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキー設定
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("エラー: GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
    sys.exit(1)

genai.configure(api_key=api_key)

#--- 利用可能なモデルをチェック ---
def get_available_model():
    print("利用可能なモデルを検索中・・・")
    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

        # 優先順位リスト
        priorities = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-lite"
            "models/gemini-1.0-pro"
        ]

        for p in priorities:
            if p in available_models:
                print(f"モデル '{p}' を使用します。")
                return p
        if available_models:
            print(f"優先モデルが見つからないため、'{available_models[0]}'を使用します。")
            return available_models[0]

        raise Exception("利用可能なモデルが見つかりません。APIキーを確認してください。")
    except Exception as e:
        print(f"モデルリストの取得中にエラーが発生しました： {e}")
        # フォールバック
        return "models/gemini-1.5-flash"

MODEL_NAME = "models/gemini-1.5-flash"
OUTPUT_FILE = "data.json"
QUESTIONS_FILE = "questions.txt"

# リトライ設定
MAX_RETRIES = 4
RETRY_WAIT_SECONDS = 50 # Flashモデルはリミットが緩いため短縮
WAIT_BETWEEN_REQUESTS = 30

print(f"=== Geminiデータ生成モード (Model: {MODEL_NAME}) ===")

# --- 1. 既存のデータを読み込む ---
existing_results = []
processed_prompts = set()
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                existing_results = json.loads(content)
                for record in existing_results:
                    if 'prompt' in record:
                        processed_prompts.add(record['prompt'])
        if existing_results:
            print(f"'{OUTPUT_FILE}' から {len(existing_results)} 件の既存データを読み込みました。")
    except Exception as e:
        print(f"警告: '{OUTPUT_FILE}' の読み込みに失敗しました。 ({e})")

# --- 2. 質問リストを読み込む ---
try:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        all_questions = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"エラー: '{QUESTIONS_FILE}' が見つかりません。")
    sys.exit(1)

questions_to_process = [q for q in all_questions if q not in processed_prompts]

if not questions_to_process:
    print("すべての質問の処理が完了しています。")
    sys.exit(0)

# --- 4. データ生成 ---
# システムインストラクション: 1.5 Flashの無料枠(1M TPM)に収まるよう、適度な長さを指定
SYSTEM_INSTRUCTION = (
"あなたはプログラミングの専門家です。質問に対して、正確かつ簡潔に回答してください。"
"回答は必ず日本語で行い、重要なポイントとコード例を含めてください。"
"全体の文字数は、句読点を含めて400文字程度に収めてください。"
)

model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION)

pbar = tqdm(questions_to_process, desc="生成中")
for prompt in pbar:
    success = False
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024, # 400文字程度なら十分
                )
            )
            
            if response.text:
                answer = response.text.strip()
                existing_results.append({"prompt": prompt, "response": answer})
                
                # 毎回保存（安全のため）
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=4)
                
                success = True
                break
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e): # Rate limit
                print(f"\n[制限エラー] APIの利用制限に達しました。")
                print(f"エラー内容： {e}")
                print("現在までのデータを保存して終了します。明日以降に再開するか、別のモデルを試してください。")
                # ループを抜けて保存処理へ
                success = False
                break
            else:
                pbar.write(f"エラー ({prompt[:20]}...): {e}")
                break

    if success:
        time.sleep(WAIT_BETWEEN_REQUESTS)

print(f"\n完了! 合計 {len(existing_results)} 件のデータを '{OUTPUT_FILE}' に保存しました。")