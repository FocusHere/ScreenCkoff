@echo off
title Wyckoff Phase Screener 80s
color 0A
echo ===================================================
echo   WYCKOFF SCREENER - INIZIALIZZAZIONE AMBIENTE
echo ===================================================
echo.

if not exist venv (
    echo Creazione ambiente virtuale Python...
    python -m venv venv
)

echo Attivazione ambiente e verifica dipendenze...
call venv\Scripts\activate
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

cls
echo ===================================================
echo   AVVIO WYCKOFF SCREENER 
echo ===================================================
echo.
python wyckoff_screener.py

pause
