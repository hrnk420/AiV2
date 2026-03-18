import google.generativeai as genai
import os
import json
import time
import sys
import random
import re
from tqdm import tqdm
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキー設定
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("エラー: GEMINI_API_KEY が設定されていません。")
    print(".env ファイルを作成し、GEMINI_API_KEY=your_key を記述してください。")
    sys.exit(1)

# --- 設定 ---
# 【重要】REST通信を使用し、モデルを固定
genai.configure(api_key=api_key, transport='rest')
MODEL_NAME = "models/gemma-3-4b-it" 
OUTPUT_FILE = "data.json"
QUESTIONS_FILE = "questions.txt"

# --- 負荷に配慮した「安全・継続」設定 ---
FIXED_WAIT = 25              # 25秒（TPM制限を考慮）
MAX_RETRIES = 3
RETRY_DELAY = 60             # エラー時は1分待機

print(f"=== データ生成モード (続きから再開対応) ===")
print(f"Model: {MODEL_NAME}")

# --- 1. 既存データの読み込み ---
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
                        processed_prompts.add(record['prompt'].strip())
        print(f"既存データ: {len(existing_results)} 件")
    except Exception as e:
        print(f"警告: 既存データの読み込みに失敗しました ({e})")

# --- 2. 質問リストの読み込み ---
try:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        all_questions = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"エラー: {QUESTIONS_FILE} が見つかりません。")
    sys.exit(1)

questions_to_process = [q for q in all_questions if q not in processed_prompts]
print(f"未処理: {len(questions_to_process)} 件 (全 {len(all_questions)} 件中)\n")

if not questions_to_process:
    print("すべての質問が処理済みです。")
    sys.exit(0)

# --- 3. データ生成 ---
SYSTEM_INSTRUCTION = (
    "あなたは優秀なPython講師です。初心者にもわかりやすく、具体的で実践的なコード例を交えて回答してください。\n"
    "回答は必ず以下の形式を守ってください。\n\n"
    "---START---\n"
    "RESPONSE: (ここに回答内容を記載)\n"
    "---END---"
)

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)

pbar = tqdm(questions_to_process, desc="生成進捗")
for original_prompt in pbar:
    success = False
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                original_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.3)
            )
            
            if response.text:
                text = response.text.strip()
                # RESPONSE: 以降を抽出
                response_match = re.search(r"RESPONSE:\s*(.*?)(?:\s*---END---|(?:\s*$))", text, re.DOTALL)
                
                if response_match:
                    r_val = response_match.group(1).strip()
                else:
                    # パターンにマッチしない場合は全文を採用
                    r_val = text.replace("---START---", "").replace("---END---", "").replace("RESPONSE:", "").strip()
                
                if r_val:
                    res_data = {"prompt": original_prompt, "response": r_val}
                    existing_results.append(res_data)
                    processed_prompts.add(original_prompt)
                    
                    # 1件ごとに保存（安全策）
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(existing_results, f, ensure_ascii=False, indent=4)
                    
                    success = True
                    break
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str:
                pbar.write(f"\n⚠ 制限(429)検知。{RETRY_DELAY}秒停止します。")
                time.sleep(RETRY_DELAY)
            else:
                pbar.write(f"\n✗ エラー: {error_str[:100]}")
                time.sleep(10)

    if not success:
        pbar.write(f"\n❌ 失敗: '{original_prompt[:20]}...' (リトライ上限)")
    
    time.sleep(FIXED_WAIT)

print(f"\n✅ 完了! 合計: {len(existing_results)} 件")
