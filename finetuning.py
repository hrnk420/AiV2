# モデル学習 (Phi-2版・GPU/4bit量子化対応)
import torch
import os
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# プロンプトテンプレート (Phi-2 形式)
PROMPT_TEMPLATE = "Instruct: {instruction}\nOutput: {response}"

def main():
    model_path = "microsoft/phi-2"
    output_dir = "./lora_output_phi2"
    
    print(f"Loading model: {model_path}...")

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = (device == "cuda")
    print(f"Using device: {device}")

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
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0} if is_cuda else None, 
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    
    if not is_cuda:
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
    if not os.path.exists("data.json"):
        print("エラー: data.json が見つかりません。先に datacreate.py を実行してください。")
        sys.exit(1)

    dataset = load_dataset("json", data_files="data.json")

    def tokenize_fn(examples):
        full_texts = [
            PROMPT_TEMPLATE.format(instruction=p, response=r) 
            for p, r in zip(examples["prompt"], examples["response"])
        ]
        tokenized = tokenizer(
            full_texts,
            max_length=512, # 1024から512に短縮してメモリ節約（必要に応じて調整）
            truncation=True,
        )
        return tokenized

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])

    # チェックポイントの有無確認
    resume_from_checkpoint = False
    if os.path.exists(output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(output_dir)):
        resume_from_checkpoint = True
        print(f"Found checkpoint in {output_dir}. Resuming training...")

    # 学習設定
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10, 
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=is_cuda, # GPU時は高速化
        max_grad_norm=0.3,
        warmup_steps=20,
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("学習を開始します...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 最終モデルの保存
    model.save_pretrained(output_dir)
    print(f"学習が完了しました。保存先: {output_dir}")

if __name__ == "__main__":
    main()
