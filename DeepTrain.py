# 1. Gerekli Kütüphanelerin Kurulumu
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

import torch
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from google.colab import drive

# ------------------------------------------------------------------------
# DRIVE BAĞLANTISI VE KLASÖR AYARLARI
# ------------------------------------------------------------------------
drive.mount('/content/drive')

# Checkpointlerin kaydedileceği YENİ ana klasör (İsim 'Deep' olarak güncellendi)
drive_output_dir = "/content/drive/MyDrive/NLPlora/DeepSon"

if not os.path.exists(drive_output_dir):
    os.makedirs(drive_output_dir)

print(f"DIKKAT: Tüm checkpointler şu klasörde saklanacak: {drive_output_dir}")
# ------------------------------------------------------------------------

# AYARLAR (Dataset ismi güncellendi)
DATASET_NAME = "Naholav/CodeGen-Deep-5K"
max_seq_length = 1024
dtype = None
load_in_4bit = False

# 2. Modelin ve Tokenizer'in Yuklenmesi
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. LoRA Adaptor Ayarlari
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0.3,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Veri Hazirligi (SADECE 'solution' KOLONU AYARLANDI)
system_prompt = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

def formatting_prompts_func(examples):
    # Input için instruction veya input'a bakmaya devam etsin (genelde instruction olur)
    inputs = examples["instruction"] if "instruction" in examples else examples["input"]

    # Sadece 'solution' kolonunu çekiyor.
    solutions = examples["solution"]

    texts = []
    for input_text, solution_text in zip(inputs, solutions):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": solution_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

# Veriyi yüklüyoruz
full_dataset = load_dataset(DATASET_NAME, split="train")

# Split kontrolü
if "split" in full_dataset.column_names:
    print("Veri seti 'split' kolonuna gore (train/test) ayriliyor...")
    train_dataset = full_dataset.filter(lambda example: example['split'] == 'train')
    eval_dataset = full_dataset.filter(lambda example: example['split'] == 'test')
else:
    print("Split kolonu bulunamadı, otomatik %90 train / %10 test ayrılıyor...")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

print(f"Dataset Ayrildi ({DATASET_NAME}) -> Train Sayisi: {len(train_dataset)}, Test Sayisi: {len(eval_dataset)}")

# Prompt formatlamasını uyguluyoruz
train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)

# 5. Egitim Konfigurasyonu
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        output_dir = drive_output_dir,

        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),

        # --- Checkpoint ve Kayıt Ayarları ---
        eval_strategy = "steps",
        eval_steps = 100,
        save_strategy = "steps",
        save_steps = 100,
        logging_steps = 20,

        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,

        save_total_limit = None,

        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
    ),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

print(f"Eğitim başlıyor... Hedef klasör: {drive_output_dir}")

# 6. Egitimi Baslat
trainer_stats = trainer.train()

# 7. Final Modeli Kaydetme (Dosya ismi 'deep' olarak güncellendi)
final_model_path = os.path.join(drive_output_dir, "final_lora_model_deep")
print(f"Eğitim bitti. En iyi final model şu adrese kaydediliyor: {final_model_path}")

model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)