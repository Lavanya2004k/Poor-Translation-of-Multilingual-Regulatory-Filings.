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

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    return text

def segment_text_into_articles(text):
    """
    Segments a block of legal text into articles, clauses, and other logical blocks.
    This function uses a multi-stage approach to be more robust.
    """
    segments = []

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
        return segments 
    print("Warning: No specific articles or clauses found. Trying to segment by general headings.")
    
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

def init_cache():
    """Initializes the SQLite cache database."""
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
    
    c.execute("PRAGMA table_info(translations)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'context_hash' not in columns:
        print("Upgrading database schema to include context_hash column...")
        c.execute("ALTER TABLE translations ADD COLUMN context_hash TEXT DEFAULT ''")
       
        c.execute("""
            CREATE TABLE translations_new (
                article_id TEXT,
                source_text TEXT,
                translated_text TEXT,
                context_hash TEXT DEFAULT '',
                PRIMARY KEY(article_id, source_text, context_hash)
            )
        """)
        c.execute("""
            INSERT INTO translations_new (article_id, source_text, translated_text, context_hash)
            SELECT article_id, source_text, translated_text, '' FROM translations
        """)
        c.execute("DROP TABLE translations")
        c.execute("ALTER TABLE translations_new RENAME TO translations")
        print("Database schema upgraded successfully.")
    
    conn.commit()
    conn.close()

def get_cached_translation(article_id, source_text, context_hash=""):
    """Retrieves a translation from the cache."""
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "SELECT translated_text FROM translations WHERE article_id=? AND source_text=? AND context_hash=?",
        (article_id, source_text, context_hash)
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def cache_translation(article_id, source_text, translated_text, context_hash=""):
    """Saves a translation to the cache."""
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?)",
        (article_id, source_text, translated_text, context_hash)
    )
    conn.commit()
    conn.close()

def create_train_val_split(translated_file, src_lang_code, tgt_lang_code, train_path="train.json", val_path="val.json", split_ratio=0.9):
    """
    Creates training and validation datasets from a single translated JSON file.
    """
    with open(translated_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        src_text = item.get("text") or item.get("original_text")
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
    """
    Fine-tunes the mBART model on the provided training data.
    """
    print("Starting fine-tuning...")
    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Set the special tokens correctly
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    def preprocess(examples):
        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        
        # Tokenize inputs with source language setting
        tokenizer.src_lang = src_lang
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
        
        # Tokenize targets with target language setting
        tokenizer.src_lang = tgt_lang  # Set to target for labels
        labels = tokenizer(targets, max_length=512, truncation=True, padding=True)
        
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

class FilingTranslatorAgent:
    """
    An agent that orchestrates the entire translation pipeline from
    ingestion to translated output with advanced error handling.
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
        
        # Critical: Set the source language on the tokenizer
        self.tokenizer.src_lang = self.src_lang
        print(f"Set tokenizer source language to: {self.src_lang}")
        print(f"Target language: {self.tgt_lang}")

    def _detect_language(self, text, expected_lang):
        """
        Simple heuristic to detect if text is in expected language.
        """
        if expected_lang == "de_DE":
            # Common German words and patterns
            german_indicators = ["der", "die", "das", "und", "ist", "sind", "für", "mit", "auf", "von", "zu", "ein", "eine", "einem", "einer", "sich", "nicht", "werden", "wurden", "wurde", "dass", "wenn", "aber", "oder", "auch", "nach", "bei", "über", "unter", "während", "wegen", "trotz", "gemäß", "§", "Abs.", "HGB", "GmbH"]
            word_count = len(text.split())
            german_matches = sum(1 for word in german_indicators if word in text.lower())
            return german_matches / max(word_count, 1) > 0.1
            
        elif expected_lang == "en_XX":
            # Common English words
            english_indicators = ["the", "and", "is", "are", "for", "with", "on", "of", "to", "a", "an", "in", "at", "by", "from", "that", "this", "these", "those", "will", "would", "could", "should", "may", "might", "can", "shall"]
            word_count = len(text.split())
            english_matches = sum(1 for word in english_indicators if word in text.lower())
            return english_matches / max(word_count, 1) > 0.1
            
        return False

    def _has_repeated_phrases(self, text, n=3):
        """Checks for excessive repetition in the translated text."""
        tokens = text.split()
        if len(tokens) < n:
            return False
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return len(ngrams) != len(set(ngrams))

    def _validate_translation_quality(self, source_text, translated_text, expected_target_lang):
        """
        Validates translation quality with improved heuristics.
        """
        issues = []
        
        # Check if translation is actually in target language
        if not self._detect_language(translated_text, expected_target_lang):
            issues.append("wrong_language")
        
        # Check if translation is too similar to source (potential copy)
        if expected_target_lang != self.src_lang:
            common_words = set(source_text.lower().split()) & set(translated_text.lower().split())
            similarity_ratio = len(common_words) / max(len(translated_text.split()), 1)
            if similarity_ratio > 0.7:  # Too many common words
                issues.append("too_similar_to_source")
        
        if self._has_repeated_phrases(translated_text, 3):
            issues.append("repetition")
        
        ratio = len(translated_text) / len(source_text) if len(source_text) > 0 else 0
        if ratio < 0.2 or ratio > 4.0: 
            issues.append("length_mismatch")
        
        if not re.search(r'[.!?]$', translated_text.strip()):
            if len(translated_text) > 50:  
                issues.append("incomplete")
        
        if len(translated_text.strip()) < 10 and len(source_text) > 50:
            issues.append("too_short")
        
        return issues

    def _translate_with_fallbacks(self, text, context_before="", context_after=""):
        """
        Translates a single text segment with multiple fallbacks for robustness.
        """
        # Ensure tokenizer is set to source language
        self.tokenizer.src_lang = self.src_lang
        
        base_gen_kwargs = dict(
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,  
            do_sample=False, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            # Add decoder_start_token_id to ensure proper target language generation
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]
        )
        
        print(f"  Translating from {self.src_lang} to {self.tgt_lang}")
        print(f"  Forced BOS token ID: {self.tokenizer.lang_code_to_id[self.tgt_lang]}")
        
        # Strategy 1: Direct translation with proper language tokens
        try:
            # Prepare input with explicit language prefix if needed
            input_text = text
            
            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            print(f"  Input tokens shape: {inputs['input_ids'].shape}")
            print(f"  First few input token IDs: {inputs['input_ids'][0][:10].tolist()}")
            
            with torch.no_grad():
                generated_tokens = self.model.generate(**inputs, **base_gen_kwargs)
            
            print(f"  Generated tokens shape: {generated_tokens.shape}")
            print(f"  First few generated token IDs: {generated_tokens[0][:10].tolist()}")
            
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
            
            print(f"  Raw translation: {translated_text[:100]}...")
            
            quality_issues = self._validate_translation_quality(text, translated_text, self.tgt_lang)
            
            if not quality_issues or len(quality_issues) == 1 and "length_mismatch" in quality_issues:
                return translated_text
            else:
                print(f"  [Strategy 1] Direct translation has issues: {quality_issues}")
        
        except Exception as e:
            print(f"  [Strategy 1] Direct translation failed: {e}")
        
        # Strategy 2: Try with more aggressive settings
        try:
            print(f"  [Strategy 2] Trying with more aggressive generation settings")
            
            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            
            aggressive_gen_kwargs = base_gen_kwargs.copy()
            aggressive_gen_kwargs.update({
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'repetition_penalty': 1.5,
                'length_penalty': 1.2
            })
            
            with torch.no_grad():
                generated_tokens = self.model.generate(**inputs, **aggressive_gen_kwargs)
            
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
            
            quality_issues = self._validate_translation_quality(text, translated_text, self.tgt_lang)
            
            if len(quality_issues) <= 1:
                return translated_text
            else:
                print(f"  [Strategy 2] Aggressive translation has issues: {quality_issues}")
        
        except Exception as e:
            print(f"  [Strategy 2] Aggressive translation failed: {e}")
        
        # Strategy 3: Sentence-by-sentence if text is long
        if len(text) > 200:
            try:
                print(f"  [Strategy 3] Trying sentence-by-sentence translation")
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if len(sentences) > 1:
                    translated_sentences = []
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:
                            self.tokenizer.src_lang = self.src_lang
                            inputs = self.tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True)
                            sent_gen_kwargs = base_gen_kwargs.copy()
                            sent_gen_kwargs['max_length'] = 256
                            
                            with torch.no_grad():
                                generated_tokens = self.model.generate(**inputs, **sent_gen_kwargs)
                            
                            sent_translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
                            translated_sentences.append(sent_translation)
                    
                    final_translation = " ".join(translated_sentences)
                    quality_issues = self._validate_translation_quality(text, final_translation, self.tgt_lang)
                    
                    if len(quality_issues) <= 2:  # More lenient for sentence-by-sentence
                        return final_translation
            
            except Exception as e:
                print(f"  [Strategy 3] Sentence-by-sentence translation failed: {e}")
        
        # Final fallback: Return a clear error message
        print(f"  [Final Fallback] All translation strategies failed")
        return f"[TRANSLATION_ERROR: Unable to translate from {self.src_lang} to {self.tgt_lang}]"

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
            
        if not text:
            print("No text extracted from the document. Exiting.")
            return

        articles = segment_text_into_articles(text)
       
        # Process long segments
        processed_articles = []
        for article in articles:
            source_text = article.get("text")
            segment_id = article.get("segment_id")
            
            if len(source_text) > 800: 
                print(f"Segment {segment_id} is very long ({len(source_text)} chars), splitting carefully.")
                
                natural_breaks = re.split(r'(?<=\.|;|\n)\s*(?=[A-Z]|\d+\.)', source_text)
                
                if len(natural_breaks) > 1:
                    current_sub_segment = ""
                    sub_segment_id_counter = 1
                    
                    for break_text in natural_breaks:
                        if len(current_sub_segment) + len(break_text) < 600:  
                            current_sub_segment += " " + break_text if current_sub_segment else break_text
                        else:
                            if current_sub_segment:
                                processed_articles.append({
                                    "segment_id": f"{segment_id}-{sub_segment_id_counter}",
                                    "text": current_sub_segment.strip()
                                })
                                sub_segment_id_counter += 1
                            current_sub_segment = break_text
                    
                    if current_sub_segment:
                        processed_articles.append({
                            "segment_id": f"{segment_id}-{sub_segment_id_counter}",
                            "text": current_sub_segment.strip()
                        })
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', source_text)
                    current_sub_segment = ""
                    sub_segment_id_counter = 1
                    
                    for sentence in sentences:
                        if len(current_sub_segment) + len(sentence) < 600:
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
        for i, article in enumerate(tqdm(processed_articles, desc="Translating")):
            article_id = article.get("segment_id", "unknown")
            source_text = article.get("text")
            if not source_text:
                continue
            
            # Simplified caching without context for now to focus on translation quality
            cached = get_cached_translation(article_id, source_text, "")
            if cached and not cached.startswith("[TRANSLATION_ERROR"):
                translated = cached
                print(f"Using cached translation for segment {article_id}")
            else:
                print(f"Translating segment {article_id} ({len(source_text)} chars)")
                print(f"Source text preview: {source_text[:100]}...")
                translated = self._translate_with_fallbacks(source_text)
                cache_translation(article_id, source_text, translated, "")

            results.append({
                "segment_id": article_id,
                "original_text": source_text,
                "translated_text": translated,
                "source_language": self.src_lang,
                "target_language": self.tgt_lang
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Translations saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mBART Fine-Tune + Translate Agent with Caching")
    
    parser.add_argument("--train_only", action="store_true", help="Only perform fine-tuning, do not translate.")
    parser.add_argument("--translated_blocks", type=str, help="Path to existing translated JSON to split into train/val")
    parser.add_argument("--train_file", type=str, help="Path to training dataset (JSON)")
    parser.add_argument("--val_file", type=str, help="Path to validation dataset (JSON)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    parser.add_argument("--translate_file", type=str, help="Path to a document (PDF/TXT) to translate.")
    parser.add_argument("--output_file", type=str, default="translated_output.json", help="Where to save translations")
    parser.add_argument("--model_path", type=str, help="Path to a pre-trained or fine-tuned model.")
    
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code (e.g., de_DE)")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code (e.g., en_XX)")
    
    args = parser.parse_args()
    init_cache()

    # Training phase
    if not args.translate_file and not args.model_path:
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

    # Translation phase
    if args.translate_file:
        model_to_use = args.model_path if args.model_path else MODEL_DIR
        
        if not os.path.exists(model_to_use):
            raise FileNotFoundError(f"Model not found at {model_to_use}. Please train a model or provide a valid --model_path.")
            
        translator_agent = FilingTranslatorAgent(model_to_use, args.src_lang, args.tgt_lang)
        translator_agent.ingest_and_translate(args.translate_file, args.output_file)