import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def main():
    lora_model_path = "./lora_output"
    base_model_path = "D:/aiV2/base_model"
    offload_folder = "D:/aiV2_offload"

    # オフロード先がなければ作成
    if not os.path.exists(offload_folder):
        os.makedirs(offload_folder)

    # ── トークナイザー ──
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # EOSトークンをPADトークンとして設定（MPT系モデル対応）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── CPU + オフロードでベースモデルロード ──
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float32,
        device_map={"": "cpu"},
        offload_folder=offload_folder
    )

    # ── LoRA適用 ──
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        dtype=torch.float32,
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
            print("⚠️ 入力が空です。文字列を確認してください。")
            continue

        # 推論
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=50
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n=== モデル出力 ===")
        print(text)

if __name__ == "__main__":
    main()
