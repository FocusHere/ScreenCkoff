@echo off
title Wyckoff Screener
cd /d "%~dp0"

echo ===============================================
echo            WYCKOFF PHASE SCREENER
echo ===============================================
echo.

REM Controlla se Python e' installato
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python non trovato. Installa Python da https://python.org
    echo e assicurati di selezionare "Add Python to PATH" durante l'installazione.
    pause
    exit /b 1
)

REM Installa le dipendenze se mancanti (rapido se gia' presenti)
pip show yfinance >nul 2>&1
if errorlevel 1 (
    echo Installazione dipendenze in corso, attendere...
    pip install -r requirements.txt
)

echo.
python wyckoff_screener.py --file tickers.txt

echo.
pause
