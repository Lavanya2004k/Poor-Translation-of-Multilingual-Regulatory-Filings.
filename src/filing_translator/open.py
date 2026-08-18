import os
import json
import time
from tqdm import tqdm
from langdetect import detect
from openai import OpenAI

client = OpenAI()

# Language mapping
def detect_language(lang_code):
    lang_map = {
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "en": "English"
    }
    return lang_map.get(lang_code.lower(), "Unknown")

# Load translation cache if it exists
cache_file = "translation_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        translation_cache = json.load(f)
else:
    translation_cache = {}

# GPT-based translation
def translate_with_gpt(segment_text, source_lang):
    prompt = f"""
You are a legal translation assistant. Your job is to translate the following {source_lang} regulatory text to fluent, accurate, and semantically correct English. Preserve article numbers, clause structure, and references exactly.

Translate this:
\"\"\"{segment_text}\"\"\"

Only return the translated text.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f" Error: {e}")
        return "[Translation failed]"

# Directories
segment_dir = "/Users/kdnai_kothalavanya/Desktop/Poor%20Translation%20of%20Multilingual%20Regulatory%20Filings_Lavanya/segmented_blocks"
output_dir = "translated_segments"
os.makedirs(output_dir, exist_ok=True)

# Process files
for filename in os.listdir(segment_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(segment_dir, filename)
        print(f" Translating {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            segments = json.load(f)

        for segment in tqdm(segments):
            if "translated_text" not in segment or segment["translated_text"] == "[Translation failed]":
                lang = detect_language(segment.get("language", ""))
                input_text = segment.get("original_text", segment.get("text", None))

                if not input_text:
                    segment["translated_text"] = "[Missing original text]"
                elif input_text in translation_cache:
                    segment["translated_text"] = translation_cache[input_text]
                else:
                    translated = translate_with_gpt(input_text, lang)
                    segment["translated_text"] = translated
                    translation_cache[input_text] = translated
                    time.sleep(1.2)  # throttle API calls

        # Save updated segments
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f" Saved translated segments to: {output_path}")

# Save updated cache
with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(translation_cache, f, ensure_ascii=False, indent=2)

print(" All translations complete with caching.")
