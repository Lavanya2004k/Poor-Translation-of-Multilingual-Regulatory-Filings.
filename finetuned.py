import os
import json
import random
import argparse
import sqlite3
import torch
from tqdm import tqdm
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import PyPDF2
import re
import gc

MODEL_DIR = "fine_tuned_mbart_legal"
CACHE_DB = "translation_cache.db"

# ====================================================================================================
# MEMORY OPTIMIZATION UTILITIES
# ====================================================================================================

def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_optimal_batch_size():
    """Get optimal batch size based on available memory"""
    if torch.backends.mps.is_available():
        # For MPS, start with smaller batch sizes
        return 2
    elif torch.cuda.is_available():
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8:
            return 2
        elif gpu_memory < 16:
            return 4
        else:
            return 8
    else:
        return 4  # CPU fallback

# ====================================================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ====================================================================================================

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def segment_text_into_articles(text):
    segments = []
    current_heading = None
    current_content = ""

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        heading_match = re.match(r'^(\d+(\.\d+)*)\s+.*', line)
        if heading_match:
            if current_heading and current_content:
                segments.append({
                    "segment_id": current_heading,
                    "text": current_content.strip()
                })
            current_heading = heading_match.group(1)
            current_content = line
        else:
            current_content += " " + line

    if current_heading and current_content:
        segments.append({
            "segment_id": current_heading,
            "text": current_content.strip()
        })

    if not segments:
        paragraphs = re.split(r'\n{2,}', text)
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:
                segments.append({
                    "segment_id": f"p{i+1}",
                    "text": paragraph
                })

    return segments

# ====================================================================================================
# CACHING FUNCTIONS
# ====================================================================================================

def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            article_id TEXT,
            source_text TEXT,
            translated_text TEXT,
            PRIMARY KEY(article_id, source_text)
        )
    """)
    conn.commit()
    conn.close()

def get_cached_translation(article_id, source_text):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "SELECT translated_text FROM translations WHERE article_id=? AND source_text=?",
        (article_id, source_text)
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def cache_translation(article_id, source_text, translated_text):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    # Fix: Explicitly specify columns to match the 3 columns in the table definition.
    c.execute(
        "INSERT OR REPLACE INTO translations (article_id, source_text, translated_text) VALUES (?, ?, ?)",
        (article_id, source_text, translated_text)
    )
    conn.commit()
    conn.close()

# ====================================================================================================
# TRAINING FUNCTIONS
# ====================================================================================================

def create_train_val_split(translated_file, src_lang_code, tgt_lang_code, train_path="train.json", val_path="val.json", split_ratio=0.9):
    with open(translated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        src_text = item.get("text") or item.get("original_text")
        tgt_text = item.get("translated_text")
        if src_text and tgt_text:
            # Truncate very long texts to save memory
            if len(src_text) > 400:
                src_text = src_text[:400] + "..."
            if len(tgt_text) > 400:
                tgt_text = tgt_text[:400] + "..."
            
            formatted_data.append({
                "translation": {
                    src_lang_code: src_text,
                    tgt_lang_code: tgt_text
                }
            })

    print(f"Total samples found for splitting: {len(formatted_data)}")

    if not formatted_data:
        print("Warning: No valid samples found in translated file for splitting.")
        return

    random.shuffle(formatted_data)
    split_index = int(len(formatted_data) * split_ratio)

    train_data = formatted_data[:split_index]
    val_data = formatted_data[split_index:]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Created {train_path} ({len(train_data)} samples) and {val_path} ({len(val_data)} samples)")

def train_mbart(train_file, val_file, src_lang, tgt_lang, epochs=3, batch_size=None, lr=3e-5):
    """Memory-optimized training function"""
    print("Starting memory-optimized fine-tuning...")
    
    # Clear memory before starting
    clear_memory()
    
    # Use optimal batch size if not specified
    if batch_size is None:
        batch_size = get_optimal_batch_size()
    
    print(f"Using batch size: {batch_size}")
    
    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    def preprocess(examples):
        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        
        # Use shorter max_length to save memory
        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=True)
        labels = tokenizer(targets, max_length=256, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = load_dataset("json", data_files=train_file)["train"]
    val_dataset = load_dataset("json", data_files=val_file)["train"]

    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

    # Memory-optimized training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4 if batch_size <= 2 else 2,  # Accumulate gradients to simulate larger batch
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=False,  # Disable FP16 for MPS stability
        bf16=False,  # Disable BF16 for MPS stability
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch",
        dataloader_pin_memory=False,  # Disable pin memory for MPS
        remove_unused_columns=True,
        dataloader_num_workers=0,  # Reduce to 0 for MPS stability
        max_grad_norm=1.0,  # Gradient clipping
        warmup_ratio=0.1,
        save_safetensors=True,
        optim="adamw_torch"  # Use standard AdamW
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model, 
        padding=True,
        max_length=256,
        pad_to_multiple_of=None  # Disable padding optimization for memory
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Add callback to clear memory between epochs
    from transformers import TrainerCallback
    
    class MemoryCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            clear_memory()
        
        def on_step_end(self, args, state, control, **kwargs):
            # Clear memory every 50 steps
            if state.global_step % 50 == 0:
                clear_memory()

    trainer.add_callback(MemoryCallback())

    try:
        trainer.train()
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"Model saved to {MODEL_DIR}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Out of memory error occurred. Try reducing batch size further.")
            print(f"Current batch size: {batch_size}")
            print("Try running with --batch_size 1 or use CPU training with --force_cpu")
            raise
        else:
            raise

# ====================================================================================================
# TRANSLATION FUNCTIONS
# ====================================================================================================

def translate_articles(articles, src_lang_code, tgt_lang_code, model_path, output_file="translated_articles.json"):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer.src_lang = src_lang_code

    results = []
    for article in tqdm(articles, desc="Translating"):
        article_id = article.get("segment_id", "unknown")
        source_text = article.get("text")
        if not source_text:
            continue

        cached = get_cached_translation(article_id, source_text)
        if cached:
            translated = cached
        else:
            inputs = tokenizer(source_text, return_tensors="pt", max_length=256, truncation=True)
            with torch.no_grad():  # Save memory during inference
                generated_tokens = model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
                    max_length=256,
                    num_beams=2,  # Reduce beam search for memory
                    early_stopping=True
                )
            translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            cache_translation(article_id, source_text, translated)

        results.append({
            "segment_id": article_id,
            "original_text": source_text,
            "translated_text": translated
        })
        
        # Clear memory periodically
        if len(results) % 50 == 0:
            clear_memory()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Translations saved to {output_file}")

# ====================================================================================================
# AGENT CLASS
# ====================================================================================================

class FilingTranslatorAgent:
    def __init__(self, model_path, src_lang, tgt_lang):
        self.model_path = model_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        clear_memory()
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_path)
        self.tokenizer.src_lang = self.src_lang

    def ingest_and_translate(self, input_file, output_file):
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension == '.pdf':
            text = extract_text_from_pdf(input_file)
        elif file_extension == '.txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_extension == '.json':
            with open(input_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            text = "\n".join([item.get("original_text", "") for item in text_data])
        else:
            raise ValueError("Unsupported file type. Provide PDF, TXT, or JSON.")

        articles = segment_text_into_articles(text)

        processed_articles = []
        for article in articles:
            source_text = article.get("text")
            segment_id = article.get("segment_id")

            # Use smaller chunks for memory efficiency
            if len(source_text) > 800:
                sentences = re.split(r'(?<=[.!?])\s+', source_text)
                current_sub_segment = ""
                sub_segment_id_counter = 1
                for sentence in sentences:
                    if len(current_sub_segment) + len(sentence) < 300:  # Smaller chunks
                        current_sub_segment += " " + sentence if current_sub_segment else sentence
                    else:
                        if current_sub_segment:
                            processed_articles.append({
                                "segment_id": f"{segment_id}-{sub_segment_id_counter}",
                                "text": current_sub_segment.strip()
                            })
                            sub_segment_id_counter += 1
                        current_sub_segment = sentence
                if current_sub_segment:
                    processed_articles.append({
                        "segment_id": f"{segment_id}-{sub_segment_id_counter}",
                        "text": current_sub_segment.strip()
                    })
            else:
                processed_articles.append(article)

        print(f"Found {len(processed_articles)} semantic blocks after processing.")

        results = []
        for article in tqdm(processed_articles, desc="Translating"):
            article_id = article.get("segment_id", "unknown")
            source_text = article.get("text")
            if not source_text:
                continue

            cached = get_cached_translation(article_id, source_text)
            if cached:
                translated = cached
            else:
                inputs = self.tokenizer(source_text, return_tensors="pt", max_length=256, truncation=True)
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs, 
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                        max_length=256,
                        num_beams=2,
                        early_stopping=True
                    )
                translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                cache_translation(article_id, source_text, translated)

            results.append({
                "segment_id": article_id,
                "original_text": source_text,
                "translated_text": translated
            })
            
            # Clear memory periodically
            if len(results) % 25 == 0:
                clear_memory()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Translations saved to {output_file}")

# ====================================================================================================
# MAIN EXECUTION LOGIC
# ====================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Optimized mBART Fine-Tune + Translate Agent")

    parser.add_argument("--train_only", action="store_true", help="Only perform fine-tuning, do not translate.")
    parser.add_argument("--translated_blocks", type=str, help="Path to existing translated JSON to split into train/val")
    parser.add_argument("--train_file", type=str, help="Path to training dataset (JSON)")
    parser.add_argument("--val_file", type=str, help="Path to validation dataset (JSON)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (reduced default)")
    parser.add_argument("--batch_size", type=int, help="Batch size (auto-detected if not specified)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU training")

    parser.add_argument("--translate_file", type=str, help="Path to a document (PDF/TXT/JSON) to translate.")
    parser.add_argument("--output_file", type=str, default="translated_output.json", help="Where to save translations")
    parser.add_argument("--model_path", type=str, help="Path to a pre-trained or fine-tuned model.")

    parser.add_argument("--src_lang", type=str, required=True, help="Source language code (e.g., de_DE)")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code (e.g., en_XX)")

    args = parser.parse_args()
    
    # Force CPU if requested
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.backends.mps.is_available = lambda: False
    
    init_cache()

    if not args.translate_file and not args.model_path:
        if args.translated_blocks and (not args.train_file or not args.val_file):
            create_train_val_split(args.translated_blocks, args.src_lang, args.tgt_lang)
            args.train_file = "train.json"
            args.val_file = "val.json"

        if not args.train_file or not args.val_file:
            raise ValueError("Training requires --translated_blocks or --train_file/--val_file arguments.")

        train_mbart(args.train_file, args.val_file, args.src_lang, args.tgt_lang,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print(f"Training complete. Fine-tuned model saved to '{MODEL_DIR}'.")

    if args.train_only:
        exit()

    if args.translate_file:
        model_to_use = args.model_path if args.model_path else MODEL_DIR
        if not os.path.exists(model_to_use):
            raise FileNotFoundError(f"Model not found at {model_to_use}. Train a model or provide --model_path.")
        translator_agent = FilingTranslatorAgent(model_to_use, args.src_lang, args.tgt_lang)
        translator_agent.ingest_and_translate(args.translate_file, args.output_file)