# モデル学習 (Phi-2版・GPU/4bit量子化対応)
import torch
import subprocess
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os

# プロンプトテンプレート (Phi-2 形式)
PROMPT_TEMPLATE = "Instruct: {instruction}\nOutput: {response}"

def update_libraries():
    """実行前に必要なライブラリをアップデートする"""
    print("Checking and updating libraries...")
    try:
        # sys.executable を使用して現在の仮想環境のpipを叩く
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", 
            "transformers", "diffusers", "accelerate", "datasets>=3.0.0"
        ])
        print("Libraries updated successfully.")
    except Exception as e:
        print(f"Warning: Failed to update libraries: {e}")

def main():
    # 実行前にライブラリを更新
    update_libraries()

    model_path = "microsoft/phi-2"
    
    print(f"Loading model: {model_path}...")

    # GPU利用可否の判定
    is_cuda = torch.cuda.is_available()
    print(f"CUDA available: {is_cuda}")

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 量子化設定 (GPUがある場合のみ適用)
    bnb_config = None
    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # モデルロード
    # low_cpu_mem_usage=True を追加してCPU上での不要な初期化（Byte型エラー）を回避
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0} if is_cuda else None, 
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    
    if is_cuda:
        base_model = base_model.to("cuda")
    else:
        base_model = base_model.to("cpu") 

    # 量子化モデルの学習準備 (GPU時のみ)
    if is_cuda:
        base_model = prepare_model_for_kbit_training(base_model)

    # LoRA適用
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "fc1", "fc2"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # データセット
    dataset = load_dataset("json", data_files="data.json")

    def tokenize_fn(examples):
        full_texts = [
            PROMPT_TEMPLATE.format(instruction=p, response=r) 
            for p, r in zip(examples["prompt"], examples["response"])
        ]
        tokenized = tokenizer(
            full_texts,
            max_length=1024,
            padding="max_length",
            truncation=True,
        )
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])

    # 学習設定 (日本語能力向上のためのしっかり学習版)
    training_args = TrainingArguments(
        output_dir="./lora_output_phi2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=15, # 3から15に増やして日本語を叩き込む
        learning_rate=2e-5, # 少しだけ上げる
        logging_steps=1,
        save_strategy="epoch", # 万が一のためにエポックごとに保存
        save_total_limit=3, # 最新の3つだけ保持
        fp16=False,
        max_grad_norm=0.3,
        warmup_steps=20,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        processing_class=tokenizer,
    )

    print("GPUでの学習を開始します...")
    # チェックポイントがあればそこから再開、なければ最初から開始
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model("./lora_output_phi2")
    print("学習が完了しました。")

if __name__ == "__main__":
    main()
