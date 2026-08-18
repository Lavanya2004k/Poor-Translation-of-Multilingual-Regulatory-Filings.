import pdfplumber
import os
import json
import logging
from langdetect import detect
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

INPUT_DIR = "/Users/kdnai_kothalavanya/Desktop/Poor%20Translation%20of%20Multilingual%20Regulatory%20Filings_Lavanya/data"
OUTPUT_DIR = "segmented_blocks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_pdf(file_path, file_name):
    segments = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in tqdm(enumerate(pdf.pages), desc=f"Processing {file_name}"):
                try:
                    text = page.extract_text()
                except Exception as e:
                    print(f" Skipped page {page_num + 1} in {file_name} due to extraction error.")
                    continue

                if not text:
                    continue

                for i, block in enumerate(text.split("\n\n")):
                    cleaned = block.strip()
                    if len(cleaned) < 20:
                        continue
                    try:
                        lang = detect(cleaned)
                    except:
                        lang = "unknown"

                    segments.append({
                        "file": file_name,
                        "page": page_num + 1,
                        "segment_id": f"{file_name}_p{page_num + 1}_s{i + 1}",
                        "language": lang,
                        "text": cleaned
                    })

        # Save output JSON
        output_file = os.path.join(OUTPUT_DIR, file_name.replace(".pdf", ".json"))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f" Saved {len(segments)} segments to {output_file}")
    except Exception as e:
        print(f" Error processing {file_name}: {e}")

if __name__ == "__main__":
    for pdf_file in os.listdir(INPUT_DIR):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, pdf_file)
            preprocess_pdf(pdf_path, pdf_file)
