import os
import pandas as pd

print(">> Scarico la lista completa dello S&P 500...")

try:
    # Scarica direttamente il dataset CSV con tutti i componenti dello S&P 500
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)

    # Formatta i ticker per Yahoo Finance (es. BRK.B -> BRK-B)
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    # Salva nella cartella corrente
    file_path = os.path.join(os.getcwd(), "tickers.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        for t in tickers:
            f.write(f"{t}\n")

    print(f"✅ COMPLETATO! Inseriti {len(tickers)} ticker in 'tickers.txt'.")

except Exception as e:
    print(f"❌ ERRORE: {e}")
