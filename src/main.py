import sys
import os
from data import load_and_merge
from preprocessing import clean_target, remove_leakage, handle_missing_values, encode_categoricals
from model import train_and_evaluate
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    # 1. carregar e mergea datasets
    df = load_and_merge()

    # 2. Limpar target
    df = clean_target(df)

    # 3. Remover leakage columns
    df = remove_leakage(df)

    # 4. Tratar valores nulos
    df = handle_missing_values(df)

    # 5. Separar X e y
    y = df["target"]
    X = df.drop(columns=["target", "delivery_status"])

    # 6. Tratar variáveis categóricas
    X = encode_categoricals(X)

    # 7. Treino, teste e avaliação
    train_and_evaluate(X, y)

