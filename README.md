# Poor Translation of Multilingual Regulatory Filings

A multilingual Agent regulatory filing translation project built around mBART-50, PDF/text extraction, translation caching, fine-tuning workflows, and a Streamlit interface for reviewing translated legal and financial document segments.

The project is intended for experimentation with translation quality in high-stakes regulatory filings, where terminology, structure, completeness, and consistency matter.

## Features

- Translate regulatory filing content from PDF or TXT files.
- Segment legal documents into articles, clauses, headings, and paragraphs.
- Cache translations locally with SQLite to avoid repeated model calls.
- Fine-tune mBART-50 on domain-specific translated examples.
- Multi agents automation 
- Run an interactive Streamlit app for upload, translation, fine-tuning, and analytics.

## Project Structure

```text
.
├── assets/                    # Static assets such as the project logo
├── configs/                   # Model and generation configuration JSON files
├── data/                      # Training and validation datasets
├── outputs/                   # Sample/generated translation outputs
├── src/
│   └── filing_translator/     # Main application and translation scripts
│       ├── complete.py        # Main mBART training and translation CLI
│       ├── finetnow.py        # Memory-optimized fine-tuning workflow
│       ├── finetuned.py       # Fine-tuned translation workflow
│       ├── frontend.py        # Streamlit web application
│       ├── open.py            # OpenAI-assisted translation workflow
│       ├── preprocess.py      # PDF preprocessing and segmentation
│       └── translatoragent.py # Translator agent implementation
├── complete.py                # Backward-compatible CLI entry point
├── frontend.py                # Backward-compatible Streamlit entry point
├── requirements.txt           # Python dependencies
└── README.md
```

Root-level Python files are lightweight compatibility wrappers. The main source files live in `src/filing_translator/`.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Lavanya2004k/Poor-Translation-of-Multilingual-Regulatory-Filings..git
cd Poor-Translation-of-Multilingual-Regulatory-Filings.
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run frontend.py
```

The app supports document translation, fine-tuning, translation cache analytics, and result downloads.

## Command-Line Translation

Use the main CLI entry point:

```bash
python complete.py \
  --translate_file path/to/document.pdf \
  --src_lang de_DE \
  --tgt_lang en_XX \
  --model_path facebook/mbart-large-50 \
  --output_file outputs/translated_output.json
```

If no `--model_path` is provided, the script expects a local fine-tuned model directory named `fine_tuned_mbart_legal`.

## Fine-Tuning

Fine-tune mBART with prepared training and validation files:

```bash
python complete.py \
  --train_file data/train.json \
  --val_file data/val.json \
  --src_lang de_DE \
  --tgt_lang en_XX \
  --epochs 5 \
  --batch_size 8
```

For lower-memory machines, use the memory-optimized workflow:

```bash
python finetnow.py \
  --train_file data/train.json \
  --val_file data/val.json \
  --src_lang de_DE \
  --tgt_lang en_XX
```

## Preprocessing PDFs

`src/filing_translator/preprocess.py` extracts PDF text into segmented JSON blocks. Update the input/output paths in that script before running it against your local PDF collection.

```bash
python preprocess.py
```

## Data

Training and validation files are stored in `data/`. Translation outputs are stored in `outputs/`. Runtime-generated artifacts such as local model checkpoints, logs, and translation cache files are ignored by Git.

## Language Codes

The mBART workflows use language codes such as:

- `en_XX` for English
- `fr_FR` for French
- `de_DE` for German
- `es_XX` for Spanish
- `it_IT` for Italian

## Notes

- Large model files are not committed to the repository.
- The SQLite cache is treated as a local runtime artifact.
- Regulatory translations should be reviewed by qualified legal, financial, or compliance professionals before official use.

## Disclaimer

This project is for research and educational purposes only. It does not provide legal, financial, regulatory, or certified translation advice.

