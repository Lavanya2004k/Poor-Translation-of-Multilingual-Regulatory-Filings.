import os
import json
import random
import argparse
import sqlite3
import torch
import evaluate
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

MODEL_DIR = "fine_tuned_mbart_legal"
CACHE_DB = "translation_cache.db"

# ====================================================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ====================================================================================================

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def segment_text_into_articles(text):
    """
    Segments a block of legal text into articles, clauses, and other logical blocks.
    This function uses a multi-stage approach to be more robust.
    """
    segments = []

    # First, try to find specific "Article" or "Clause" patterns
    pattern_specific = re.compile(r'(?:Article|Clause)\s+(\d+\.?\d*)\s+(.*?)(?=\n(?:Article|Clause|\Z))', re.DOTALL)
    matches_specific = list(pattern_specific.finditer(text))
    
    if matches_specific:
        for match in matches_specific:
            segment_id = match.group(1)
            segment_text = match.group(2).strip()
            if segment_text:
                segments.append({
                    "segment_id": segment_id,
                    "text": segment_text
                })
        return segments # Return if successful with specific pattern

    # If specific patterns are not found, try a more general heading pattern
    print("Warning: No specific articles or clauses found. Trying to segment by general headings.")
    
    # This logic iterates through lines and builds segments based on numbered headings.
    current_heading = None
    current_content = ""
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if the line is a new heading (e.g., "1", "1.1", "2.2.1")
        heading_match = re.match(r'^(\d+(\.\d+)*)\s+.*', line)
        if heading_match:
            # If we have a previous heading and content, save it
            if current_heading and current_content:
                segments.append({
                    "segment_id": current_heading,
                    "text": current_content.strip()
                })
            
            # Start a new segment
            current_heading = heading_match.group(1)
            # The content for this heading starts on the same line
            current_content = line
        else:
            # Append non-heading line to current content
            current_content += " " + line
            
    # Add the last segment after the loop ends
    if current_heading and current_content:
        segments.append({
            "segment_id": current_heading,
            "text": current_content.strip()
        })
    
    # Final fallback: split into paragraphs or chunks if no other segmentation works
    if not segments:
        print("Warning: No headers found. Splitting text into paragraphs.")
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
    c.execute(
        "INSERT OR REPLACE INTO translations VALUES (?, ?, ?)",
        (article_id, source_text, translated_text)
    )
    conn.commit()
    conn.close()

# ====================================================================================================
# TRANSLATION AND TRAINING FUNCTIONS
# ====================================================================================================

def create_train_val_split(translated_file, src_lang_code, tgt_lang_code, train_path="train.json", val_path="val.json", split_ratio=0.9):
    with open(translated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        src_text = item.get("text")
        tgt_text = item.get("translated_text")
        if src_text and tgt_text:
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

def train_mbart(train_file, val_file, src_lang, tgt_lang, epochs=5, batch_size=8, lr=5e-5):
    print("Starting fine-tuning...")
    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    def preprocess(examples):
        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        labels = tokenizer(targets, max_length=512, truncation=True, text_target=targets)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = load_dataset("json", data_files=train_file)["train"]
    val_dataset = load_dataset("json", data_files=val_file)["train"]

    tokenized_train = train_dataset.map(preprocess, batched=True)
    tokenized_val = val_dataset.map(preprocess, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

def translate_articles(articles, src_lang_code, tgt_lang_code, model_path, output_file="translated_articles.json"):
    print(f"Loading model from {model_path}...")
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
            inputs = tokenizer(source_text, return_tensors="pt", max_length=512, truncation=True)
            generated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code]
            )
            translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            cache_translation(article_id, source_text, translated)

        results.append({
            "segment_id": article_id,
            "original_text": source_text,
            "translated_text": translated
        })

    with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Translations saved to {output_file}")

# ====================================================================================================
# AGENT CLASS
# ====================================================================================================

class FilingTranslatorAgent:
    """
    An agent that orchestrates the entire translation pipeline from
    ingestion to translated output.
    """
    def __init__(self, model_path, src_lang, tgt_lang):
        self.model_path = model_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Loads the tokenizer and model."""
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_path)
        self.tokenizer.src_lang = self.src_lang

    def ingest_and_translate(self, input_file, output_file):
        """
        Ingests a document, segments it, translates each segment, and
        saves the output.
        """
        print(f"Ingesting and processing {input_file}...")
        file_extension = os.path.splitext(input_file)[1].lower()

        if file_extension == '.pdf':
            text = extract_text_from_pdf(input_file)
        elif file_extension == '.txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file type. Please provide a PDF or TXT file.")
            
        # Segment the text into articles/clauses
        articles = segment_text_into_articles(text)
        
        # New logic to handle long segments
        processed_articles = []
        for article in articles:
            source_text = article.get("text")
            segment_id = article.get("segment_id")
            
            # Simple check for length; 1000 characters is a rough estimate for MBart's 512 tokens
            if len(source_text) > 1000:
                print(f"Segment {segment_id} is very long, splitting into sub-segments.")
                
                # Split by sentences (using periods, etc.)
                sentences = re.split(r'(?<=[.!?])\s+', source_text)
                
                current_sub_segment = ""
                sub_segment_id_counter = 1
                for sentence in sentences:
                    # Append sentences until we hit a size limit
                    if len(current_sub_segment) + len(sentence) < 500: # 500 characters is a safe limit
                        current_sub_segment += " " + sentence if current_sub_segment else sentence
                    else:
                        # Save the current chunk and start a new one
                        if current_sub_segment:
                            processed_articles.append({
                                "segment_id": f"{segment_id}-{sub_segment_id_counter}",
                                "text": current_sub_segment.strip()
                            })
                            sub_segment_id_counter += 1
                        current_sub_segment = sentence
                
                # Add the last sub-segment
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

            # Check cache for existing translation (Memory-based inference)
            cached = get_cached_translation(article_id, source_text)
            if cached:
                translated = cached
            else:
                inputs = self.tokenizer(source_text, return_tensors="pt", max_length=512, truncation=True)
                generated_tokens = self.model.generate(
                    **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]
                )
                translated = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                cache_translation(article_id, source_text, translated)

            results.append({
                "segment_id": article_id,
                "original_text": source_text,
                "translated_text": translated
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Translations saved to {output_file}")

# ====================================================================================================
# MAIN EXECUTION LOGIC
# ====================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mBART Fine-Tune + Translate Agent with Caching")
    
    # Arguments for training
    parser.add_argument("--train_only", action="store_true", help="Only perform fine-tuning, do not translate.")
    parser.add_argument("--translated_blocks", type=str, help="Path to existing translated JSON to split into train/val")
    parser.add_argument("--train_file", type=str, help="Path to training dataset (JSON)")
    parser.add_argument("--val_file", type=str, help="Path to validation dataset (JSON)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    # Arguments for translation
    parser.add_argument("--translate_file", type=str, help="Path to a document (PDF/TXT) to translate.")
    parser.add_argument("--output_file", type=str, default="translated_output.json", help="Where to save translations")
    parser.add_argument("--model_path", type=str, help="Path to a pre-trained or fine-tuned model.")
    
    # Common arguments
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code (e.g., de_DE)")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code (e.g., en_XX)")
    
    args = parser.parse_args()
    init_cache()

    # --- Training Logic ---
    if not args.translate_file and not args.model_path:
        # If no model is provided and no file to translate, we assume the goal is to train.
        if args.translated_blocks and (not args.train_file or not args.val_file):
            create_train_val_split(args.translated_blocks, args.src_lang, args.tgt_lang)
            args.train_file = "train.json"
            args.val_file = "val.json"
        
        if not args.train_file or not args.val_file:
            raise ValueError("Training requires --translated_blocks or --train_file/--val_file arguments.")
        
        train_mbart(args.train_file, args.val_file, args.src_lang, args.tgt_lang, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print("Training complete. A fine-tuned model has been saved to the 'fine_tuned_mbart_legal' directory.")
        
    if args.train_only:
        exit()

    # --- Translation Logic ---
    if args.translate_file:
        model_to_use = args.model_path if args.model_path else MODEL_DIR
        
        if not os.path.exists(model_to_use):
            raise FileNotFoundError(f"Model not found at {model_to_use}. Please train a model or provide a valid --model_path.")
            
        translator_agent = FilingTranslatorAgent(model_to_use, args.src_lang, args.tgt_lang)
        translator_agent.ingest_and_translate(args.translate_file, args.output_file) 