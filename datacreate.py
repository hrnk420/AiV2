import google.generativeai as genai
import os
import json
import time
import sys
import random # randomモジュールを追加

# --- カウントダウンタイマー関数 ---
def countdown_timer(seconds):
    """指定された秒数をカウントダウン表示する関数"""
    try:
        for i in range(seconds, 0, -1):
            mins, secs = divmod(i, 60)
            timer_display = f"    ...次の処理まであと {mins:02d}分{secs:02d}秒..."
            sys.stdout.write(timer_display)
            sys.stdout.flush()
            time.sleep(1)
            sys.stdout.write('\r' + ' ' * len(timer_display) + '\r')
    except KeyboardInterrupt:
        print("\nカウントダウンが中断されました。")
        raise
    print("\n    ...待機完了。処理を再開します...")

# APIキー設定
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "models/gemini-2.5-flash"
output_file = "data.json"
questions_file = "questions.txt"

# リトライ設定
RETRY_WAIT_SECONDS = 30
MAX_RETRIES = 3

# 動的待機時間調整の設定
MIN_WAIT_TIME = 180  # 最小待機時間 (3分)
MAX_WAIT_TIME = 1800 # 最大待機時間 (30分)
WAIT_TIME_DECREASE_FACTOR = 0.9 # 成功時に待機時間を減らす係数
WAIT_TIME_INCREASE_FACTOR = 1.5 # 失敗時に待機時間を増やす係数

print("=== Geminiデータ生成モード (再開・タイマー・リトライ・動的待機機能付き) ===")

# --- 1. 既存のデータを読み込む ---
existing_results = []
processed_prompts = set()
if os.path.exists(output_file):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            if os.path.getsize(output_file) > 0:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    print(f"警告: '{output_file}' の形式が不正なため、初期化します。")
                    existing_results = []
            
            for record in existing_results:
                if 'prompt' in record:
                    processed_prompts.add(record['prompt'])
        if existing_results:
            print(f"'{output_file}' から {len(existing_results)} 件の既存データを読み込みました。")
    except (json.JSONDecodeError, IOError) as e:
        print(f"警告: '{output_file}' の読み込みに失敗しました。ファイルを初期化します。 ({e})")
        existing_results = []
        processed_prompts = set()

# --- 2. 質問リストを読み込む ---
try:
    with open(questions_file, "r", encoding="utf-8") as f:
        all_questions = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"エラー: '{questions_file}' が見つかりません。")
    exit()

# --- 3. 未処理の質問を特定する ---
questions_to_process = [q for q in all_questions if q not in processed_prompts]

if not questions_to_process:
    print("すべての質問の処理が完了しています。")
    exit()

print(f"全 {len(all_questions)} 問中、未処理の {len(questions_to_process)} 問を処理します。")

# --- 4. 未処理の質問を処理し、結果を追記 ---
newly_added_count = 0
total_to_process = len(questions_to_process)

# 動的待機時間調整の状態変数
current_wait_time = MIN_WAIT_TIME
consecutive_successes = 0
consecutive_failures = 0

for count, prompt in enumerate(questions_to_process, 1):
    model = genai.GenerativeModel(model_name)
    print(f"[{len(existing_results) + count}/{len(all_questions)}] 🧠 '{prompt[:30]}...' を生成中...")
    
    # プロンプトに文字数制限の指示を強化
    modified_prompt = f"{prompt}。回答は厳密に100文字以内に完結させてください。"
    
    # 質問内容に応じてmax_output_tokensを動的に設定
    max_tokens_for_this_request = 2048 if "コード" in prompt or "python" in prompt.lower() else 1024

    response = None # responseを初期化
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                modified_prompt, # 修正したプロンプトを使用
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=max_tokens_for_this_request # 動的に設定した値を使用
                ),
                request_options={"timeout": 180} # タイムアウトを180秒に変更
            )
            break # 成功したらリトライループを抜ける
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    ⚠️ エラー: {e}。{RETRY_WAIT_SECONDS}秒後に再試行します… (試行 {attempt + 1}/{MAX_RETRIES})")
                countdown_timer(RETRY_WAIT_SECONDS) # リトライ待機もカウントダウン表示
            else:
                print(f"    ❌ エラー: {e}。最大試行回数 ({MAX_RETRIES}) を超えました。この質問の処理をスキップします。")
                response = None # 最終的に失敗した場合はresponseをNoneにする

    # リトライ成功した場合、または最終的に失敗した場合の処理
    try:
        if response:
            if response.candidates and response.candidates[0].content.parts:
                answer = response.text.strip()
                record = {"prompt": prompt, "response": answer}
                existing_results.append(record)
                newly_added_count += 1
                print(f"    ✅ 生成完了")

                # 成功した場合のみ待機処理を実行
                if count < total_to_process:
                    # 成功時の待機時間調整
                    consecutive_successes += 1
                    consecutive_failures = 0
                    current_wait_time = max(MIN_WAIT_TIME, current_wait_time * WAIT_TIME_DECREASE_FACTOR)
                    countdown_timer(int(current_wait_time)) 
            else:
                reason = response.candidates[0].finish_reason.name if response.candidates else 'N/A'
                print(f"    ⚠️ 応答が生成されませんでした。理由: {reason}")
                # 応答がない場合も、次の処理まで少し待つ
                if count < total_to_process:
                    # 失敗時の待機時間調整
                    consecutive_failures += 1
                    consecutive_successes = 0
                    current_wait_time = min(MAX_WAIT_TIME, current_wait_time * WAIT_TIME_INCREASE_FACTOR)
                    countdown_timer(int(current_wait_time)) # 失敗時は長めの待機
        else:
            print(f"    ⚠️ この質問の生成は最終的に失敗しました。スキップします。")
            # 失敗した場合も、次の処理まで少し待つ
            if count < total_to_process:
                # 失敗時の待機時間調整
                consecutive_failures += 1
                consecutive_successes = 0
                current_wait_time = min(MAX_WAIT_TIME, current_wait_time * WAIT_TIME_INCREASE_FACTOR)
                countdown_timer(int(current_wait_time)) # 失敗時は長めの待機

    except KeyboardInterrupt:
        print("\n処理が中断されました。現在までの結果を保存します。")
        break
    except Exception as e:
        print(f"    ⚠️ 予期せぬエラー: {e}")
        print("処理を中断し、現在までの結果を保存します。")
        break
    finally:
        # ループの反復の最後に、成功・失敗・中断(break前)に関わらず必ず保存
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"    ⚠️ ファイルの書き込みに失敗しました: {e}")

print("\n" + "="*30)
if newly_added_count > 0:
    print(f"🎉 今回 {newly_added_count} 件の新しい結果を追加し、合計 {len(existing_results)} 件を '{output_file}' に保存しました。")
else:
    print("新しい結果は追加されませんでした。")
print("="*30)