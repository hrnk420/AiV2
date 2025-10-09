import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import time

def main():
    lora_model_path = "./lora_output"
    base_model_path = "./mpt-7b-instruct"
    offload_folder = "D:/aiV2_offload"

    # オフロード先がなければ作成
    if not os.path.exists(offload_folder):
        os.makedirs(offload_folder)

    # ── トークナイザー ──
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # EOSトークンをPADトークンとして設定（MPT系モデル対応）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── CPU + オフロードでベースモデルロード ──
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map={"": "cpu"},
        offload_folder=offload_folder
    )

    # ── LoRA適用 ──
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        offload_folder=offload_folder
    )
    model.eval()

    print("=== 推論モード（CPU＋LoRA＋オフロード） ===")
    print("終了するには 'exit' と入力してください")

    while True:
        prompt = input("\n入力してください: ")
        if prompt.lower() == "exit":
            break

        # トークナイズ（安全パディング）
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cpu")
        if inputs["input_ids"].size(1) == 0:
            print("Warning: 入力が空です。文字列を確認してください。")
            continue

        # 推論
        print("\n⏳ 応答を生成中です。CPUでの処理には数分かかることがあります。お待ちください...")
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50
                )
            end_time = time.time()
            duration = end_time - start_time
            print(f"    (生成時間: {duration:.2f}秒)")

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\n=== モデル出力 ===")
            print(text)

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"\nError: 応答の生成中にエラーが発生しました。(処理時間: {duration:.2f}秒)")
            print(f"    詳細: {e}")
            continue

if __name__ == "__main__":
    main()
