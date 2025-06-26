import argparse
import json
import os
from datetime import datetime
from torch.utils.data import Dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# === 新增：图片resize函数 ===
from PIL import Image

def resize_and_copy_image(src_path, dst_path, size=(640, 480)):
    try:
        img = Image.open(src_path)
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path)
        return dst_path
    except Exception as e:
        print(f"Warning: Failed to resize {src_path} due to {e}")
        return src_path  # fallback: return original path if failed

def resize_images_in_jsonl(input_jsonl, output_jsonl, resized_img_root):
    """
    遍历jsonl，把所有图片路径的图片resize后保存到resized_img_root下，并替换jsonl里路径为新文件
    """
    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for line in infile:
            sample = json.loads(line.strip())
            changed = False

            # 遍历messages，找到所有 type:image
            for message in sample.get("messages", []):
                for content in message.get("content", []):
                    if content.get("type") == "image":
                        orig_path = content["image"]
                        # 生成新图片路径（保持相对结构）
                        rel_path = os.path.relpath(orig_path, '/')
                        new_img_path = os.path.join(resized_img_root, rel_path)
                        content["image"] = resize_and_copy_image(orig_path, new_img_path)
                        changed = True
            outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Resized images and updated dataset saved to {output_jsonl}")

def convert_to_conversation(sample):
    """
    Converts a single sample (json) into the required format with instructions and messages.
    """
    instruction = sample["messages"][0]["content"]
    image_contents = [{"type": "image", "image": img} for img in sample["images"]]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction}] + image_contents
            
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

    # 1. 数据转换
    processed_jsonl = os.path.join(args.output_dir, "processed.jsonl")
    process_jsonl_files(args.jsonl, processed_jsonl)
    args.jsonl = processed_jsonl

    # 2. 图片resize并路径替换
    resized_img_root = "/your/custom/path/resized_images"  # 修改图片保存路径
    resized_jsonl = "/your/custom/path/processed_resized.jsonl"  # 修改压缩后的JSON地址
    resize_images_in_jsonl(args.jsonl, resized_jsonl, resized_img_root)
    args.jsonl = resized_jsonl

    # 3. 训练
    train(args)