# Wyckoff Phase Screener

Screener Python che analizza titoli via Yahoo Finance (`yfinance`) e individua
4 fasi in sequenza obbligata, ispirate al metodo Wyckoff:

1. **Support Base** (Accumulation) – lateralità, range stretto, supporto testato più volte
2. **Over Limit** (Breakout) – rottura sopra la resistenza, con volume alto
3. **Discovery** (Markup) – trend rialzista, RSI alto, volumi sostenuti
4. **Distribution** – appiattimento dei massimi + divergenza ribassista RSI

## Installazione (una tantum)

1. Installa Python da https://python.org (spunta "Add Python to PATH" durante l'installazione)
2. Metti tutti i file di questa cartella insieme (`wyckoff_screener.py`, `requirements.txt`,
   `avvia_screener.bat`, `tickers.txt`)
3. Apri il Prompt dei comandi in quella cartella ed esegui una volta:
   ```
   pip install -r requirements.txt
   ```

## Uso rapido

- Modifica `tickers.txt` inserendo un ticker per riga (es. `AAPL`, `ENI.MI` per Borsa Italiana, ecc.)
- Fai doppio click su `avvia_screener.bat`

Oppure da terminale:
```
python wyckoff_screener.py --tickers AAPL,MSFT,NVDA --period 1y
python wyckoff_screener.py --file tickers.txt --period 2y --output risultati.csv
```

## Aggiungerlo al menu Start di Windows

1. Tasto destro su `avvia_screener.bat` → **Crea collegamento**
2. Sposta il collegamento in:
   `C:\Users\<TuoNome>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs`
3. Il launcher comparirà cercandolo nel menu Start ("Wyckoff Screener" o simile,
   rinomina pure il collegamento come preferisci)

*(In alternativa, se vuoi un vero file .exe con icona personalizzata senza bisogno
di Python installato sul PC che lo usa, si può creare con `pyinstaller` — dimmelo
e prepariamo anche quello.)*

## Parametri regolabili

Tutte le soglie (ampiezza range, RSI minimo, moltiplicatore volume, ecc.) sono
nella classe `Config` in cima a `wyckoff_screener.py`. Modificale per adattare
la sensibilità del rilevamento al tuo mercato/timeframe.

## Output

Lo script stampa a schermo la fase attuale rilevata per ogni titolo e salva un
CSV completo (default `wyckoff_results.csv`) con date di inizio/fine di ogni
fase individuata nell'ultimo ciclo trovato.

## Note

- `yfinance` non è un'API ufficiale Yahoo: in rari casi può richiedere un
  aggiornamento della libreria se Yahoo cambia qualcosa lato loro.
- Serve connessione internet quando lo script gira.
- Con molti ticker, lo script scarica i dati in sequenza; per liste molto
  lunghe valuta di aggiungere una pausa tra le richieste (dimmelo se ti serve).
