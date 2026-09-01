"""
Wyckoff Phase Screener - 80s Retro Hex / Cyberpunk Edition
==========================================================
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)

try:
    import yfinance as yf
except ImportError:
    print(f"{Fore.RED}ERRORE: la libreria 'yfinance' non e' installata.")
    sys.exit(1)

try:
    from scipy.signal import argrelextrema
except ImportError:
    print(f"{Fore.RED}ERRORE: la libreria 'scipy' non e' installata.")
    sys.exit(1)


# =====================================================================
# BANNER ANNI '80 / ASCII ART
# =====================================================================

BANNER = f"""{Fore.CYAN}
░█▀█░█▀█░█▀▀░█▀▀░█▀█░░░█▀▀░█▀▀░█▀▄░█▀▀░█▀▀░█▀█░█▀▀░█▀▄
░█▀▀░█▀█░█▀▀░█▀▀░█░█░░░▀▀█░█░░░█▀▄░█▀▀░█▀▀░█░█░█▀▀░█▀▄
░▀░░░▀░▀░▀░░░▀░░░▀▀▀░░░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀░▀
                                                                                              
{Fore.MAGENTA}======================================================================
 [0x57][0x59][0x43][0x4B][0x4F][0x46][0x46] -- SYSTEM v2.0 (1984)
======================================================================{Style.RESET_ALL}
"""

@dataclass
class Config:
    base_window: int = 20
    base_range_pct_max: float = 0.10
    base_sma_slope_max: float = 0.02
    base_min_touches: int = 2
    base_touch_tolerance: float = 0.02

    breakout_margin: float = 0.02
    breakout_volume_mult: float = 1.5
    breakout_lookahead: int = 15

    discovery_rsi_min: float = 60.0
    discovery_window: int = 15
    discovery_volume_mult: float = 1.1

    distribution_window: int = 15
    distribution_flatten_pct: float = 0.03
    distribution_rsi_drop: float = 5.0

    sma_short: int = 20
    sma_long: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_sma: int = 20


CFG = Config()


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


@dataclass
class PhaseEvent:
    phase: str
    start: pd.Timestamp
    end: pd.Timestamp
    details: dict = field(default_factory=dict)


def detect_support_bases(df: pd.DataFrame, cfg: Config = CFG) -> List[PhaseEvent]:
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
            i += w
        else:
            i += 1

    return events


def detect_breakout(df: pd.DataFrame, base: PhaseEvent, cfg: Config = CFG) -> Optional[PhaseEvent]:
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
    start_idx = df.index.get_loc(breakout.end)
    end_idx = min(start_idx + 90, len(df) - 1)
    if end_idx - start_idx < cfg.discovery_window:
        return None

    window = df.iloc[start_idx:end_idx + 1]
    vol_avg_overall = df["Volume_SMA"].iloc[start_idx]

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
    start_idx = df.index.get_loc(discovery.end)
    end_idx = min(start_idx + 60, len(df) - 1)
    if end_idx - start_idx < cfg.distribution_window:
        return None

    window = df.iloc[start_idx:end_idx + 1]
    closes = window["Close"].values
    rsis = window["RSI"].values

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
    if not cycles:
        return "nessuna fase"
    last = cycles[-1]
    order = ["distribution", "discovery", "over_limit", "support_base"]
    for phase in order:
        if phase in last:
            return phase
    return "nessuna fase"


def fetch_top_100_large_caps() -> List[str]:
    print(f"{Fore.CYAN}>> BUS INIT: Searching Top 100 Cap (> $12B)...")
    try:
        q = yf.EquityQuery('and', [
            yf.EquityQuery('gt', ['intradaymarketcap', 12_000_000_000]),
            yf.EquityQuery('eq', ['region', 'us'])
        ])
        scr = yf.Screener()
        scr.set_body({'size': 100, 'offset': 0, 'sortField': 'intradaymarketcap', 'sortAsc': False, 'query': q.to_dict()})
        res = scr.response
        tickers = [item['symbol'] for item in res.get('quotes', [])]
        if tickers:
            return tickers[:100]
    except Exception:
        pass
    
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        "AVGO", "PEP", "KO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT"
    ]


def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def screen_ticker(ticker: str, cfg: Config = CFG, period: str = "1y") -> dict:
    df = fetch_data(ticker, period=period)
    if df is None or len(df) < cfg.sma_long + cfg.base_window:
        return {"ticker": ticker, "status": "dati insufficienti", "passed": False, "current_phase": "N/D"}

    df = add_indicators(df, cfg)
    cycles = detect_wyckoff_cycles(df, cfg)
    phase = current_phase_label(cycles)

    complete_cycles = sum(1 for c in cycles if "distribution" in c)
    passed = phase != "nessuna fase"

    return {
        "ticker": ticker,
        "status": "ok",
        "passed": passed,
        "current_phase": phase,
        "cicli_rilevati": len(cycles),
        "cicli_completi": complete_cycles,
        "ultimo_prezzo": round(float(df["Close"].iloc[-1]), 2),
    }


def screen_tickers(tickers: List[str], cfg: Config = CFG, period: str = "1y") -> pd.DataFrame:
    rows = []
    
    # Barra personalizzata stile retrò
    pbar = tqdm(
        tickers,
        desc=f"{Fore.MAGENTA}[0xSCANNING]{Style.RESET_ALL}",
        bar_format="{l_bar}{bar:25}{r_bar}",
        colour="cyan"
    )

    for t in pbar:
        t = t.strip().upper()
        if not t:
            continue
        pbar.set_postfix_str(f"0xHEX_ADDR: {t}")
        res = screen_ticker(t, cfg, period)
        rows.append(res)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Wyckoff Phase Screener - 80s Edition")
    parser.add_argument("--tickers", type=str, help="Lista ticker separati da virgola")
    parser.add_argument("--file", type=str, help="File .txt con un ticker per riga")
    parser.add_argument("--top100", action="store_true", help="Analizza i primi 100 titoli per Market Cap (>12B)")
    parser.add_argument("--period", type=str, default="1y", help="Periodo storico")
    parser.add_argument("--output", type=str, default="wyckoff_results.csv", help="File CSV di output")
    return parser.parse_args()


def main():
    print(BANNER)

    args = parse_args()
    tickers: List[str] = []

    if args.top100:
        tickers = fetch_top_100_large_caps()
    elif args.tickers:
        tickers.extend(args.tickers.split(","))
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            tickers.extend(line.strip() for line in f if line.strip())

    if not tickers:
        print(f"{Fore.GREEN}>> SELECT SYSTEM OPERATION:")
        print(" [0x01] EXECUTE SCAN: TOP 100 LARGE CAPS (> $12B)")
        print(" [0x02] LOAD MEMORY FILE: tickers.txt")
        choice = input(f"\n{Fore.YELLOW}ENTER OPTION (1/2): {Style.RESET_ALL}").strip()

        if choice == "1":
            tickers = fetch_top_100_large_caps()
        elif choice == "2":
            with open("tickers.txt", "r", encoding="utf-8") as f:
                tickers = [line.strip() for line in f if line.strip()]
        else:
            sys.exit(0)

    print(f"\n{Fore.GREEN}>> SYSTEM READY. STARTING ANALYSIS AT {datetime.now().strftime('%H:%M:%S')}...\n")
    
    results = screen_tickers(tickers, CFG, period=args.period)

    print(f"\n{Fore.MAGENTA}======================================================================")
    print(f"{Fore.CYAN}                      [0xOUTPUT] SCAN RESULTS                        ")
    print(f"{Fore.MAGENTA}======================================================================\n{Style.RESET_ALL}")
    
    if not results.empty:
        results["STATUS"] = results["passed"].apply(lambda x: f"{Fore.GREEN}✅ PASS" if x else f"{Fore.RED}❌ NO")
        
        cols_show = ["STATUS", "ticker", "current_phase", "ultimo_prezzo", "cicli_rilevati"]
        cols_show = [c for c in cols_show if c in results.columns]

        pd.set_option('display.max_rows', 150)
        pd.set_option('display.width', 1000)
        print(results[cols_show].to_string(index=False))

        results.drop(columns=["STATUS"], errors="ignore").to_csv(args.output, index=False)
        print(f"\n{Fore.CYAN}>> DUMP SAVED TO MEMORY: {args.output}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
