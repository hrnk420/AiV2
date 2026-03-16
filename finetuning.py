# モデル学習 (Phi-2版・GPU/4bit量子化対応)
import torch
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

def main():
    model_path = "microsoft/phi-2"
    
    print(f"Loading model: {model_path}...")

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4bit量子化の設定 (VRAM 6GB 対策)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # モデルロード (GPU明示的に指定)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=False # 標準実装を使用
    )

    # 量子化モデルの学習準備
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA適用 (Phi-2の現在の実装名に合わせる)
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
            max_length=256, 
            padding="max_length",
            truncation=True,
        )
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])

    # 学習設定 (GPU向けに最適化)
    training_args = TrainingArguments(
        output_dir="./lora_output_phi2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # メモリ節約
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        fp16=True, # GPUでの高速化
        optim="paged_adamw_8bit", # bitsandbytesの最適化
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        processing_class=tokenizer,
    )

    print("GPUでの学習を開始します...")
    trainer.train()
    trainer.save_model("./lora_output_phi2")
    print("学習が完了しました。")

if __name__ == "__main__":
    main()
