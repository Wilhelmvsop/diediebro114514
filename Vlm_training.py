import argparse
import json
import os
from datetime import datetime
from torch.utils.data import Dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

def convert_to_conversation(sample):
    """
    Converts a single sample (json) into the required format with instructions and messages.
    """
    instruction = "Write the LaTeX representation for this image."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["images"][0]}  # Adjust as needed
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["messages"][1]["content"]}]
        },
    ]
    return {"messages": conversation}

def process_jsonl_files(input_jsonl_path, output_jsonl_path):
    """
    Processes a JSONL file, converts each JSON object to the required format, and saves to a new file.
    """
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            converted = convert_to_conversation(sample)
            outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
    print(f"Converted data saved to {output_jsonl_path}")

class QwenVisionJsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "messages" in item:
                    self.samples.append({"messages": item["messages"]})
                else:
                    print("Warning: line missing 'messages' field, skipped.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train(args):
    dataset = QwenVisionJsonlDataset(jsonl_path=args.jsonl)

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc_steps,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )

    print("======== 训练开始 ========")
    FastVisionModel.for_training(model)
    stats = trainer.train()
    print("======== 训练完成 ========")
    print(stats)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_dir = os.path.join(args.output_dir, f"lora_{timestamp}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"[✓] LoRA adapters saved to {lora_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="./train_data/my_aeb.jsonl")
    parser.add_argument("--output_dir", default="./fine_train_data")
    parser.add_argument("--model_name", default="/home/k8s/coolwin/qwen2.5-vl-3B")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 强制预处理
    processed_jsonl = os.path.join(args.output_dir, "processed.jsonl")
    process_jsonl_files(args.jsonl, processed_jsonl)
    args.jsonl = processed_jsonl

    train(args)
