"""
Wyckoff Phase Screener - 80s Retro Hex / Cyberpunk Edition v2.8
==============================================================
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# BANNER RETRÒ & GUIDA INTERFACCIA
# =====================================================================

BANNER = f"""{Fore.CYAN}
░█▀█░█▀█░█▀▀░█▀▀░█▀█░░░█▀▀░█▀▀░█▀▄░█▀▀░█▀▀░█▀█░█▀▀░█▀▄
░█▀▀░█▀█░█▀▀░█▀▀░█░█░░░▀▀█░█░░░█▀▄░█▀▀░█▀▀░█░█░█▀▀░█▀▄
░▀░░░▀░▀░▀░░░▀░░░▀▀▀░░░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀░▀
                                                      
{Fore.MAGENTA}======================================================================
 [0x57][0x59][0x43][0x4B][0x4F][0x46][0x46] -- SYSTEM v2.8 (ACTIVE CYCLES FILTER)
======================================================================{Style.RESET_ALL}
"""

LEGENDA = f"""
{Fore.YELLOW}=== LEGENDA STRUTTURA & FASI WYCKOFF ==={Style.RESET_ALL}
 ┌─────────────────┬─────────────────────────────────────────────────┐
 │ Colonna         │ Descrizione                                     │
 ├─────────────────┼─────────────────────────────────────────────────┤
 │ STATUS          │ ✅ PASS = Fase attiva rilevata | ❌ NO = Assente  │
 │ TICKER          │ Simbolo del titolo analizzato                  │
 │ FASE ATTIVA     │ Fase di Wyckoff correntemente identificata      │
 │ PREZZO          │ Ultimo prezzo di chiusura disponibile ($)       │
 │ CICLI           │ Numero di cicli Wyckoff ATTIVI nel periodo      │
 └─────────────────┴─────────────────────────────────────────────────┘

 {Fore.CYAN}Fasi del ciclo:{Style.RESET_ALL}
  • {Fore.GREEN}support_base{Style.RESET_ALL}  : Accumulazione / Supporto (Compressione volatilità)
  • {Fore.YELLOW}over_limit{Style.RESET_ALL}    : Breakout confermato con incremento di volume
  • {Fore.CYAN}discovery{Style.RESET_ALL}     : Markup / Trend rialzista in espansione
  • {Fore.RED}distribution{Style.RESET_ALL}  : Distribuzione / Possibile inversione ribassista
"""

PHASE_COLORS = {
    "support_base": Fore.GREEN,
    "over_limit": Fore.YELLOW,
    "discovery": Fore.CYAN,
    "distribution": Fore.RED,
    "nessuna fase": Fore.LIGHTBLACK_EX,
    "N/D": Fore.LIGHTBLACK_EX
}


@dataclass
class Config:
    base_window: int = 30                 # Finestra bilanciata per basi di accumulazione
    base_range_pct_max: float = 0.08      # Tolleranza range 8%
    base_sma_slope_max: float = 0.02
    base_min_touches: int = 2
    base_touch_tolerance: float = 0.02

    breakout_margin: float = 0.015        # Margine breakout 1.5%
    breakout_volume_mult: float = 1.2
    breakout_lookahead: int = 20

    discovery_rsi_min: float = 50.0       # Soglia RSI per intercettare trend iniziali
    discovery_window: int = 10
    discovery_volume_mult: float = 1.0

    distribution_window: int = 15
    distribution_flatten_pct: float = 0.03
    distribution_rsi_drop: float = 4.0

    max_phase_age: int = 30               # Massimo 30 candele (~1.5 mesi) di vecchiezza per considerare il ciclo attivo

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
    while i <= n - w:
        window = df.iloc[i:i + w]
        close_mean = window["Close"].mean()
        rng_pct = (window["High"].max() - window["Low"].min()) / close_mean

        sma_start = window["SMA_short"].iloc[0]
        sma_end = window["SMA_short"].iloc[-1]
        sma_slope = abs(sma_end - sma_start) / sma_start if sma_start and not np.isnan(sma_start) else np.nan

        bw_now = window["BB_bandwidth"].iloc[-1]
        bw_ref = bandwidth_median.iloc[i + w - 1]
        compressed = (bw_now <= bw_ref * 1.1) if not np.isnan(bw_ref) else True

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
            i += 2

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
            and row["Volume"] >= vol_avg * cfg.breakout_volume_mult
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
        trend_ok = (sub["SMA_short"] >= sub["SMA_long"]).mean() > 0.5
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

    last_processed_idx = -1

    for base in bases:
        base_start_idx = df.index.get_loc(base.start)
        if base_start_idx <= last_processed_idx:
            continue

        cycle = {"support_base": base}
        last_event_end = base.end

        breakout = detect_breakout(df, base, cfg)
        if breakout:
            cycle["over_limit"] = breakout
            last_event_end = breakout.end

            discovery = detect_discovery(df, breakout, cfg)
            if discovery:
                cycle["discovery"] = discovery
                last_event_end = discovery.end

                distribution = detect_distribution(df, discovery, cfg)
                if distribution:
                    cycle["distribution"] = distribution
                    last_event_end = distribution.end

        cycles.append(cycle)
        last_processed_idx = df.index.get_loc(last_event_end)

    return cycles


def get_active_cycles_and_phase(cycles: List[dict], total_bars: int, df: pd.DataFrame, cfg: Config = CFG):
    """
    Filtra i cicli restituendo solo quelli che sono ancora attivi
    e determina la fase corrente.
    """
    if not cycles:
        return 0, "nessuna fase"

    active_cycles_count = 0
    current_phase = "nessuna fase"

    order = ["distribution", "discovery", "over_limit", "support_base"]

    for cycle in cycles:
        cycle_is_active = False
        for phase in order:
            if phase in cycle:
                event: PhaseEvent = cycle[phase]
                event_end_idx = df.index.get_loc(event.end)
                if (total_bars - 1 - event_end_idx) <= cfg.max_phase_age:
                    cycle_is_active = True
                    # Se è l'ultimo ciclo rilevato ed è attivo, cattura la fase
                    if cycle == cycles[-1]:
                        current_phase = phase
                    break
        if cycle_is_active:
            active_cycles_count += 1

    return active_cycles_count, current_phase


def fetch_all_yahoo_large_caps() -> List[str]:
    print(f"{Fore.CYAN}>> BUS INIT: Recupero lista titoli da Yahoo Finance...")
    try:
        q = yf.EquityQuery('gt', ['intradaymarketcap', 10_000_000_000])
        scr = yf.Screener()
        scr.set_body({'size': 250, 'offset': 0, 'sortField': 'intradaymarketcap', 'sortAsc': False, 'query': q.to_dict()})
        res = scr.response
        tickers = [item['symbol'] for item in res.get('quotes', [])]
        if tickers:
            print(f"{Fore.GREEN}>> Recuperati {len(tickers)} titoli per l'analisi.")
            return tickers
    except Exception:
        pass

    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        "AVGO", "PEP", "KO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
        "ORCL", "BAC", "CRM", "AMD", "NFLX", "INTC", "DIS", "PFE", "NKE", "TXN"
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
        return {
            "TICKER": ticker,
            "STATUS_BOOL": False,
            "FASE_ATTIVA": "N/D",
            "PREZZO": 0.0,
            "CICLI": 0
        }

    df = add_indicators(df, cfg)
    all_cycles = detect_wyckoff_cycles(df, cfg)
    
    # Rilevamento cicli e fasi attive
    active_cycles, phase = get_active_cycles_and_phase(all_cycles, len(df), df, cfg)
    passed = phase != "nessuna fase"

    return {
        "TICKER": ticker,
        "STATUS_BOOL": passed,
        "FASE_ATTIVA": phase,
        "PREZZO": round(float(df["Close"].iloc[-1]), 2),
        "CICLI": active_cycles
    }


def screen_tickers_parallel(tickers: List[str], cfg: Config = CFG, period: str = "1y", max_workers: int = 10) -> pd.DataFrame:
    results_map = {}

    pbar = tqdm(
        total=len(tickers),
        desc=f"{Fore.MAGENTA}[12M SCAN]{Style.RESET_ALL}",
        bar_format="{l_bar}{bar:25}{r_bar}",
        colour="cyan"
    )

    clean_tickers = [t.strip().upper() for t in tickers if t.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(screen_ticker, t, cfg, period): t
            for t in clean_tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                results_map[ticker] = data
            except Exception:
                results_map[ticker] = {
                    "TICKER": ticker,
                    "STATUS_BOOL": False,
                    "FASE_ATTIVA": "N/D",
                    "PREZZO": 0.0,
                    "CICLI": 0
                }
            
            pbar.update(1)
            pbar.set_postfix_str(f"LAST: {ticker}")

    pbar.close()

    ordered_rows = [results_map[t] for t in clean_tickers if t in results_map]
    return pd.DataFrame(ordered_rows)


def print_formatted_table(df: pd.DataFrame):
    headers = ["STATUS", "TICKER", "FASE ATTIVA", "PREZZO", "CICLI"]
    widths = [8, 8, 15, 10, 7]

    header_str = (
        f"{headers[0]:<{widths[0]}}"
        f"{headers[1]:<{widths[1]}}"
        f"{headers[2]:<{widths[2]}}"
        f"{headers[3]:>{widths[3]}}"
        f"{headers[4]:>{widths[4]}}"
    )
    print(f"{Fore.WHITE}{Style.BRIGHT}{header_str}{Style.RESET_ALL}")
    print("-" * sum(widths))

    for _, row in df.iterrows():
        status_raw = "✅ PASS" if row["STATUS_BOOL"] else "❌ NO"
        status_color = Fore.GREEN if row["STATUS_BOOL"] else Fore.RED
        status_fmt = f"{status_color}{status_raw:<{widths[0]}}{Style.RESET_ALL}"

        ticker_fmt = f"{Fore.WHITE}{row['TICKER']:<{widths[1]}}{Style.RESET_ALL}"

        fase_raw = str(row["FASE_ATTIVA"])
        fase_color = PHASE_COLORS.get(fase_raw, Fore.WHITE)
        fase_fmt = f"{fase_color}{fase_raw:<{widths[2]}}{Style.RESET_ALL}"

        prezzo_raw = f"{row['PREZZO']:.2f}"
        prezzo_fmt = f"{Fore.CYAN}{prezzo_raw:>{widths[3]}}{Style.RESET_ALL}"

        cicli_raw = str(row["CICLI"])
        cicli_color = Fore.CYAN if row["CICLI"] > 0 else Fore.LIGHTBLACK_EX
        cicli_fmt = f"{cicli_color}{cicli_raw:>{widths[4]}}{Style.RESET_ALL}"

        print(f"{status_fmt}{ticker_fmt}{fase_fmt}{prezzo_fmt}{cicli_fmt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Wyckoff Phase Screener - 80s Edition")
    parser.add_argument("--tickers", type=str, help="Lista ticker separati da virgola")
    parser.add_argument("--file", type=str, help="File .txt con un ticker per riga")
    parser.add_argument("--all", action="store_true", help="Analizza tutti i titoli reperibili su Yahoo Finance (>10B cap)")
    parser.add_argument("--period", type=str, default="1y", help="Periodo storico (default: 1y = 12 mesi)")
    parser.add_argument("--output", type=str, default="wyckoff_results.csv", help="File CSV di output")
    parser.add_argument("--threads", type=int, default=10, help="Numero di thread paralleli")
    return parser.parse_args()


def main():
    print(BANNER)
    print(LEGENDA)

    args = parse_args()
    tickers: List[str] = []

    if args.all:
        tickers = fetch_all_yahoo_large_caps()
    elif args.tickers:
        tickers.extend(args.tickers.split(","))
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            tickers.extend(line.strip() for line in f if line.strip())

    if not tickers:
        print(f"\n{Fore.GREEN}>> MODALITÀ DI SCANSIONE:")
        print(" [1] ANALISI COMPLETA TITOLI YAHOO FINANCE (> $10B Cap)")
        print(" [2] CARICA FILE MEMORIA: tickers.txt")
        choice = input(f"\n{Fore.YELLOW}SELEZIONA OPZIONE (1/2): {Style.RESET_ALL}").strip()

        if choice == "1":
            tickers = fetch_all_yahoo_large_caps()
        elif choice == "2":
            with open("tickers.txt", "r", encoding="utf-8") as f:
                tickers = [line.strip() for line in f if line.strip()]
        else:
            sys.exit(0)

    print(f"\n{Fore.GREEN}>> AVVIO ANALISI 12 MESI: {datetime.now().strftime('%H:%M:%S')} ({len(tickers)} titoli in corso)...")

    results = screen_tickers_parallel(tickers, CFG, period=args.period, max_workers=args.threads)

    print(f"\n{Fore.MAGENTA}======================================================================")
    print(f"{Fore.CYAN}                      RISULTATI ANALISI SCREENER                      ")
    print(f"{Fore.MAGENTA}======================================================================\n{Style.RESET_ALL}")

    if not results.empty:
        print_formatted_table(results)

        csv_df = results.rename(columns={
            "STATUS_BOOL": "STATUS",
            "FASE_ATTIVA": "FASE ATTIVA"
        })
        csv_df["STATUS"] = csv_df["STATUS"].apply(lambda x: "PASS" if x else "NO")
        csv_df.to_csv(args.output, index=False)

        print(f"\n{Fore.CYAN}>> REPORT SALVATO SU DISCO: {args.output}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
