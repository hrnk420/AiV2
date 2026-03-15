# モデル学習 (Phi-2版・認証不要・軽量高性能)
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# プロンプトテンプレート (Phi-2 形式)
PROMPT_TEMPLATE = "Instruct: {instruction}\nOutput: {response}"

def main():
    # 軽量で高性能な Phi-2 を使用 (約5GB)
    model_path = "microsoft/phi-2"
    
    print(f"Loading model: {model_path}...")

    # ハードウェア検知
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # モデルロード
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else {"": "cpu"},
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # LoRA適用 (Phi-2のレイヤー名に最適化)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["Wqkv", "fc1", "fc2"], 
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
            max_length=256, 
            padding="max_length",
            truncation=True,
        )
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])

    # 学習設定
    training_args = TrainingArguments(
        output_dir="./lora_output_phi2",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="no",
        fp16=(device == "cuda"),
        no_cuda=(device == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    print("学習を開始します。これには時間がかかる場合があります...")
    trainer.train()
    trainer.save_model("./lora_output_phi2")
    print("学習が完了しました。差分データは ./lora_output_phi2 に保存されました。")

if __name__ == "__main__":
    main()
