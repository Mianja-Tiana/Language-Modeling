"""
Prétraitement du corpus texte avant fine-tuning de GPT-2.

Lit un fichier CSV contenant une colonne 'text', découpe/filtre les textes
selon leur longueur, nettoie le texte (regex) et écrit le résultat dans
un fichier .txt utilisable pour l'entraînement.
"""

import re
import numpy as np
import pandas as pd


def preprocess(inputs, min_len, max_len):
    """Découpe les textes trop longs en phrases et filtre les textes trop courts."""
    new_data = []

    for txt in inputs:
        if len(txt) < min_len:
            continue
        if len(txt) < max_len:
            new_data.append(txt)
        else:
            sents = txt.split(".")
            acc = ""
            for sent in sents:
                if len(acc + sent) > max_len:
                    new_data.append(acc)
                    acc = ""
                else:
                    acc += sent
            if min_len < len(acc):
                new_data.append(acc)

    return np.array(new_data)


def cleaning(s):
    """Nettoyage basique du texte (ponctuation, chiffres, espaces, artefacts d'URL)."""
    s = str(s)
    s = re.sub(r'\s\W', ' ', s)
    s = re.sub(r'\W,\s', ' ', s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[!@#$_]', '', s)
    s = s.replace("co", "")
    s = s.replace("https", "")
    return s


def run(input_csv, output_txt, min_len=0, max_len=1024):
    df = pd.read_csv(input_csv)
    text_data = df['text'].values
    processed_data = preprocess(text_data, min_len, max_len)

    with open(output_txt, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(cleaning(item))

    print(f"{len(processed_data)} textes écrits dans {output_txt}")


if __name__ == "__main__":
    run(input_csv="data/rakitra.csv", output_txt="data/rakitra.txt")
