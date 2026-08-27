# GPT-2 Fine-Tuning — Rakitra

Fine-tuning du modèle **GPT-2** (Hugging Face `transformers`) sur un corpus de texte personnalisé (`rakitra.csv`), avec prétraitement, entraînement, et génération de texte.

## Structure du projet

```
.
├── README.md
├── requirements.txt
├── .gitignore               
├── models/                
└── src/
    ├── preprocess.py       
    ├── train.py             
    └── inference.py        
```

## Installation

```bash
pip install -r requirements.txt
```


### 3. Génération de texte

```bash
python src/inference.py
```
Le script demande un prompt et une longueur maximale, puis génère du texte avec le modèle fine-tuné.

## Notes

- Le modèle entraîné (`models/`) n'est pas inclus dans le repo (trop lourd pour GitHub). 


