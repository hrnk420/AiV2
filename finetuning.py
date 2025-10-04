#モデルセット→.jsonファイルで学習トレーニングまで（CPUオフロード版）
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import os

def main():
    model_path = "./mpt-7b-instruct"
    offload_folder = "D:/aiV2_offload"
    if not os.path.exists(offload_folder):
        os.makedirs(offload_folder)

    print(f"Loading model from local directory: {model_path}...")

    # Config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_config["attn_impl"] = "torch"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # モデルロード（CPUオフロード対応）
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},        # CPUメイン
        offload_folder=offload_folder  # 一時オフロード先
    )

    # LoRA適用
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["Wqkv"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # データセット
    dataset = load_dataset("json", data_files="data.json")

    # tokenize関数
    def tokenize_fn(examples):
        inputs = examples["prompt"]
        targets = examples["response"]

        model_inputs = tokenizer(
            inputs,
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # 学習設定（CPU前提）
    training_args = TrainingArguments(
        output_dir="./lora_output",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        no_cuda=True,  # GPUを使わない
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./lora_output")
    print("LoRA fine-tuning complete!")

if __name__ == "__main__":
    main()
