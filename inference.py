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

    print("Loading model for inference...")

    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4bit量子化の設定 (VRAM 6GB 対策)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ベースモデルロード (GPU明示的に指定)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=False
    )

    # LoRA適用
    if os.path.exists(lora_model_path):
        print(f"Loading LoRA weights from {lora_model_path}...")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
    else:
        print("Warning: 学習済みデータが見つかりません。ベースモデルのみで対話します。")
        model = base_model

    model.eval()

    print("\n=== 対話モード（終了するには 'exit' と入力） ===")

    while True:
        user_input = input("\n質問を入力してください: ")
        if user_input.lower() == "exit":
            break

        # プロンプト作成
        full_prompt = PROMPT_TEMPLATE.format(instruction=user_input)
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda") # GPUに移動
        
        print("\n⏳ 思考中...")
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
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
