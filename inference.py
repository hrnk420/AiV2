# 推論モード (Phi-2版・GPU/4bit量子化対応)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import time

# プロンプトテンプレート (Phi-2 形式)
PROMPT_TEMPLATE = "Instruct: {instruction}\nOutput: "

def main():
    lora_model_path = "./lora_output_phi2"
    base_model_path = "microsoft/phi-2"

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model for inference...")

    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 量子化設定 (GPUがある場合のみ)
    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # ベースモデルロード
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if device == "cpu":
        base_model = base_model.to("cpu")

    # LoRA適用
    if os.path.exists(lora_model_path):
        print(f"Loading LoRA weights from {lora_model_path}...")
        try:
            model = PeftModel.from_pretrained(base_model, lora_model_path)
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            print("ベースモデルのみで続行します。")
            model = base_model
    else:
        print("Warning: 学習済みデータが見つかりません。ベースモデルのみで対話します。")
        model = base_model

    model.eval()

    print("\n=== 対話モード（終了するには 'exit' と入力） ===")

    while True:
        user_input = input("\n質問を入力してください: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # プロンプト作成
        full_prompt = PROMPT_TEMPLATE.format(instruction=user_input)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        print("\n⏳ 思考中...")
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # 応答部分のみデコード
            response_ids = outputs[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            print(f"\n=== モデルの回答 (生成時間: {time.time() - start_time:.2f}秒) ===")
            print(text.strip())

        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
