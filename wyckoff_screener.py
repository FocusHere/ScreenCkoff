"""
Wyckoff Phase Screener
=======================
Rileva su una serie storica di prezzi le 4 fasi (in ordine obbligato):

    1) SUPPORT BASE   (Accumulation) - lateralita', range stretto, volatilita' compressa
    2) OVER LIMIT     (Breakout)     - rottura sopra la resistenza della base, volume alto
    3) DISCOVERY      (Markup)       - trend rialzista con momentum, RSI alto, volumi sostenuti
    4) DISTRIBUTION   (Topping)      - prezzo si appiattisce sui massimi, divergenza ribassista RSI

Le fasi devono presentarsi in questo ordine cronologico. Lo screener scansiona
una lista di ticker via Yahoo Finance (yfinance) e segnala in che fase si trova
ogni titolo, oltre a eventuali cicli completi individuati nello storico.

Uso rapido:
    python wyckoff_screener.py --tickers AAPL,MSFT,NVDA
    python wyckoff_screener.py --file tickers.txt --period 2y

Autore: generato con Claude
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERRORE: la libreria 'yfinance' non e' installata.")
    print("Installala con: pip install -r requirements.txt")
    sys.exit(1)

try:
    from scipy.signal import argrelextrema
except ImportError:
    print("ERRORE: la libreria 'scipy' non e' installata.")
    print("Installala con: pip install -r requirements.txt")
    sys.exit(1)


# =====================================================================
# CONFIGURAZIONE / SOGLIE (modificabili)
# =====================================================================

@dataclass
class Config:
    # --- Support Base ---
    base_window: int = 20              # giorni della finestra per rilevare la base
    base_range_pct_max: float = 0.10   # ampiezza massima del range (10% del prezzo medio)
    base_sma_slope_max: float = 0.02   # pendenza massima della SMA (quasi piatta)
    base_min_touches: int = 2          # numero minimo di tocchi sul livello di supporto
    base_touch_tolerance: float = 0.02 # tolleranza % per considerare un "tocco" del supporto

    # --- Over Limit (Breakout) ---
    breakout_margin: float = 0.02      # % sopra la resistenza per confermare rottura
    breakout_volume_mult: float = 1.5  # volume minimo = media * questo fattore
    breakout_lookahead: int = 15       # giorni successivi alla base in cui cercare il breakout

    # --- Discovery (Markup) ---
    discovery_rsi_min: float = 60.0    # RSI minimo medio nella fase
    discovery_window: int = 15         # giorni minimi di durata per confermare la fase
    discovery_volume_mult: float = 1.1 # volume medio minimo rispetto alla media generale

    # --- Distribution ---
    distribution_window: int = 15      # giorni della finestra per rilevare distribuzione
    distribution_flatten_pct: float = 0.03  # variazione massima dei massimi (appiattimento)
    distribution_rsi_drop: float = 5.0      # calo minimo di RSI in divergenza

    # --- Indicatori generali ---
    sma_short: int = 20
    sma_long: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_sma: int = 20


CFG = Config()


# =====================================================================
# INDICATORI TECNICI
# =====================================================================

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger_bandwidth(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / sma


def add_indicators(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(cfg.sma_short).mean()
    df["SMA_long"] = df["Close"].rolling(cfg.sma_long).mean()
    df["RSI"] = compute_rsi(df["Close"], cfg.rsi_period)
    df["ATR"] = compute_atr(df, cfg.atr_period)
    df["BB_bandwidth"] = compute_bollinger_bandwidth(df["Close"], cfg.sma_short)
    df["Volume_SMA"] = df["Volume"].rolling(cfg.volume_sma).mean()
    return df


# =====================================================================
# RILEVAMENTO FASI
# =====================================================================

@dataclass
class PhaseEvent:
    phase: str
    start: pd.Timestamp
    end: pd.Timestamp
    details: dict = field(default_factory=dict)


def detect_support_bases(df: pd.DataFrame, cfg: Config = CFG) -> List[PhaseEvent]:
    """Individua finestre di consolidamento/base (fase 1)."""
    events = []
    n = len(df)
    w = cfg.base_window
    if n < w + cfg.sma_long:
        return events

    bandwidth_median = df["BB_bandwidth"].rolling(60, min_periods=20).median()

    i = cfg.sma_long
    while i < n - w:
        window = df.iloc[i:i + w]
        close_mean = window["Close"].mean()
        rng_pct = (window["High"].max() - window["Low"].min()) / close_mean

        sma_start = window["SMA_short"].iloc[0]
        sma_end = window["SMA_short"].iloc[-1]
        sma_slope = abs(sma_end - sma_start) / sma_start if sma_start and not np.isnan(sma_start) else np.nan

        bw_now = window["BB_bandwidth"].iloc[-1]
        bw_ref = bandwidth_median.iloc[i + w - 1]
        compressed = (bw_now < bw_ref) if not np.isnan(bw_ref) else True

        support_level = window["Low"].min()
        touches = (
            (window["Low"] - support_level).abs() / support_level <= cfg.base_touch_tolerance
        ).sum()

        if (
            rng_pct <= cfg.base_range_pct_max
            and not np.isnan(sma_slope)
            and sma_slope <= cfg.base_sma_slope_max
            and compressed
            and touches >= cfg.base_min_touches
        ):
            events.append(PhaseEvent(
                phase="support_base",
                start=window.index[0],
                end=window.index[-1],
                details={
                    "support_level": round(float(support_level), 4),
                    "resistance_level": round(float(window["High"].max()), 4),
                    "range_pct": round(float(rng_pct), 4),
                    "touches": int(touches),
                },
            ))
            i += w  # salta oltre la base individuata per evitare sovrapposizioni
        else:
            i += 1

    return events


def detect_breakout(df: pd.DataFrame, base: PhaseEvent, cfg: Config = CFG) -> Optional[PhaseEvent]:
    """Cerca la rottura sopra la resistenza della base (fase 2 - Over Limit)."""
    resistance = base.details["resistance_level"]
    start_idx = df.index.get_loc(base.end)
    end_idx = min(start_idx + cfg.breakout_lookahead, len(df) - 1)

    for i in range(start_idx + 1, end_idx + 1):
        row = df.iloc[i]
        vol_avg = row["Volume_SMA"]
        if np.isnan(vol_avg) or vol_avg == 0:
            continue
        if (
            row["Close"] > resistance * (1 + cfg.breakout_margin)
            and row["Volume"] > vol_avg * cfg.breakout_volume_mult
        ):
            return PhaseEvent(
                phase="over_limit",
                start=df.index[i],
                end=df.index[i],
                details={
                    "breakout_close": round(float(row["Close"]), 4),
                    "resistance": round(float(resistance), 4),
                    "volume_ratio": round(float(row["Volume"] / vol_avg), 2),
                },
            )
    return None


def detect_discovery(df: pd.DataFrame, breakout: PhaseEvent, cfg: Config = CFG) -> Optional[PhaseEvent]:
    """Cerca la fase di markup/momentum dopo il breakout (fase 3 - Discovery)."""
    start_idx = df.index.get_loc(breakout.end)
    end_idx = min(start_idx + 90, len(df) - 1)  # osserva fino a ~90 giorni dopo
    if end_idx - start_idx < cfg.discovery_window:
        return None

    window = df.iloc[start_idx:end_idx + 1]
    vol_avg_overall = df["Volume_SMA"].iloc[start_idx]

    # cerca la sotto-finestra piu' lunga che rispetta le condizioni di markup
    best_end = None
    for j in range(cfg.discovery_window, len(window)):
        sub = window.iloc[:j]
        higher_highs = sub["Close"].iloc[-1] > sub["Close"].iloc[0]
        rsi_ok = sub["RSI"].mean() >= cfg.discovery_rsi_min
        trend_ok = (sub["SMA_short"] > sub["SMA_long"]).mean() > 0.6
        vol_ok = (
            sub["Volume"].mean() >= (vol_avg_overall or 0) * cfg.discovery_volume_mult
            if vol_avg_overall and not np.isnan(vol_avg_overall) else True
        )
        if higher_highs and rsi_ok and trend_ok and vol_ok:
            best_end = j
        else:
            if best_end is not None:
                break

    if best_end is None:
        return None

    sub = window.iloc[:best_end]
    return PhaseEvent(
        phase="discovery",
        start=sub.index[0],
        end=sub.index[-1],
        details={
            "price_change_pct": round(float(sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1) * 100, 2),
            "avg_rsi": round(float(sub["RSI"].mean()), 2),
        },
    )


def detect_distribution(df: pd.DataFrame, discovery: PhaseEvent, cfg: Config = CFG) -> Optional[PhaseEvent]:
    """Cerca appiattimento dei massimi + divergenza RSI dopo il markup (fase 4 - Distribution)."""
    start_idx = df.index.get_loc(discovery.end)
    end_idx = min(start_idx + 60, len(df) - 1)
    if end_idx - start_idx < cfg.distribution_window:
        return None

    window = df.iloc[start_idx:end_idx + 1]
    closes = window["Close"].values
    rsis = window["RSI"].values

    # massimi locali (picchi) su prezzo e RSI
    peak_idx = argrelextrema(closes, np.greater_equal, order=3)[0]
    if len(peak_idx) < 2:
        return None

    p1, p2 = peak_idx[0], peak_idx[-1]
    price_change = (closes[p2] - closes[p1]) / closes[p1]
    rsi_change = rsis[p2] - rsis[p1]

    flattening = abs(price_change) <= cfg.distribution_flatten_pct
    bearish_divergence = (closes[p2] >= closes[p1]) and (rsi_change <= -cfg.distribution_rsi_drop)

    if flattening or bearish_divergence:
        return PhaseEvent(
            phase="distribution",
            start=window.index[p1],
            end=window.index[p2],
            details={
                "price_change_pct": round(float(price_change) * 100, 2),
                "rsi_change": round(float(rsi_change), 2),
                "bearish_divergence": bool(bearish_divergence),
                "flattening": bool(flattening),
            },
        )
    return None


def detect_wyckoff_cycles(df: pd.DataFrame, cfg: Config = CFG) -> List[dict]:
    """Ricostruisce cicli completi (o parziali) rispettando l'ordine cronologico."""
    cycles = []
    bases = detect_support_bases(df, cfg)

    for base in bases:
        cycle = {"support_base": base}
        breakout = detect_breakout(df, base, cfg)
        if breakout:
            cycle["over_limit"] = breakout
            discovery = detect_discovery(df, breakout, cfg)
            if discovery:
                cycle["discovery"] = discovery
                distribution = detect_distribution(df, discovery, cfg)
                if distribution:
                    cycle["distribution"] = distribution
        cycles.append(cycle)

    return cycles


def current_phase_label(cycles: List[dict]) -> str:
    """Determina la fase piu' recente rilevata per il titolo."""
    if not cycles:
        return "nessuna fase rilevata"
    last = cycles[-1]
    order = ["distribution", "discovery", "over_limit", "support_base"]
    for phase in order:
        if phase in last:
            return phase
    return "nessuna fase rilevata"


# =====================================================================
# SCREENER PRINCIPALE
# =====================================================================

def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [!] Errore scaricando {ticker}: {e}")
        return None


def screen_ticker(ticker: str, cfg: Config = CFG, period: str = "1y") -> dict:
    df = fetch_data(ticker, period=period)
    if df is None or len(df) < cfg.sma_long + cfg.base_window:
        return {"ticker": ticker, "status": "dati insufficienti"}

    df = add_indicators(df, cfg)
    cycles = detect_wyckoff_cycles(df, cfg)
    phase = current_phase_label(cycles)

    complete_cycles = sum(1 for c in cycles if "distribution" in c)

    result = {
        "ticker": ticker,
        "status": "ok",
        "current_phase": phase,
        "cicli_rilevati": len(cycles),
        "cicli_completi": complete_cycles,
        "ultimo_prezzo": round(float(df["Close"].iloc[-1]), 2),
    }

    if cycles:
        last = cycles[-1]
        for phase_name, ev in last.items():
            result[f"{phase_name}_start"] = ev.start.strftime("%Y-%m-%d")
            result[f"{phase_name}_end"] = ev.end.strftime("%Y-%m-%d")

    return result


def screen_tickers(tickers: List[str], cfg: Config = CFG, period: str = "1y") -> pd.DataFrame:
    rows = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        print(f"Analizzo {t}...")
        rows.append(screen_ticker(t, cfg, period))
    return pd.DataFrame(rows)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Wyckoff Phase Screener")
    parser.add_argument("--tickers", type=str, help="Lista ticker separati da virgola (es. AAPL,MSFT)")
    parser.add_argument("--file", type=str, help="File .txt con un ticker per riga")
    parser.add_argument("--period", type=str, default="1y", help="Periodo storico (es. 6mo, 1y, 2y)")
    parser.add_argument("--output", type=str, default="wyckoff_results.csv", help="File CSV di output")
    return parser.parse_args()


def main():
    args = parse_args()

    tickers: List[str] = []
    if args.tickers:
        tickers.extend(args.tickers.split(","))
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            tickers.extend(line.strip() for line in f if line.strip())

    if not tickers:
        print("Nessun ticker specificato.")
        print("Uso: python wyckoff_screener.py --tickers AAPL,MSFT,NVDA")
        print("  oppure: python wyckoff_screener.py --file tickers.txt")
        entered = input("\nInserisci ticker separati da virgola: ").strip()
        if not entered:
            sys.exit(0)
        tickers = entered.split(",")

    print(f"\n=== Wyckoff Screener - {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    results = screen_tickers(tickers, CFG, period=args.period)

    print("\n--- RISULTATI ---")
    if not results.empty:
        cols_show = ["ticker", "status", "current_phase", "cicli_rilevati", "cicli_completi", "ultimo_prezzo"]
        cols_show = [c for c in cols_show if c in results.columns]
        print(results[cols_show].to_string(index=False))
        results.to_csv(args.output, index=False)
        print(f"\nRisultati completi salvati in: {args.output}")
    else:
        print("Nessun risultato.")


if __name__ == "__main__":
    main()
