# GPT-2 Fine-Tuning — Rakitra

Fine-tuning the **GPT-2** model (Hugging Face `transformers`) on a custom text corpus (`rakitra.csv`), including preprocessing, training, and text generation.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── preprocess.py
    ├── train.py
    └── inference.py
````

## Installation

```bash
pip install -r requirements.txt
```

### 3. Text Generation

```bash
python src/inference.py
```

The script prompts the user for an input prompt and a maximum length, then generates text using the fine-tuned model.

## Notes

* The trained model (`models/`) is not included in the repository because it is too large for GitHub.

```

