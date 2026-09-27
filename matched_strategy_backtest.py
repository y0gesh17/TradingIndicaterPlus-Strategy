from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    box_size: int = 12
    smoothing_len: int = 36
    filter_weekend: bool = False
    sensitivity_factor: float = 1.0
    enable_vol_imbalance: bool = True
    vol_ratio_threshold: float = 2.5
    initial_capital: float = 10_000.0
    leverage: float = 25.0
    long_sl: float = 0.985
    long_tp: float = 1.045
    short_sl: float = 1.015
    short_tp: float = 0.955


def _load_ohlcv(csv_file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()

    aliases = {
        "Open time": [
            "Open time",
            "Datetime",
            "Date",
            "Time",
            "Timestamp",
            "timestamp",
            "time",
            "date",
            "datetime",
            "open_time",
            "open time",
        ],
        "Open": ["Open", "open", "o"],
        "High": ["High", "high", "h"],
        "Low": ["Low", "low", "l"],
        "Close": ["Close", "close", "c"],
        "Volume": ["Volume", "volume", "vol", "v"],
    }

    for canonical, names in aliases.items():
        if canonical not in df.columns:
            for name in names:
                if name in df.columns:
                    df[canonical] = df[name]
                    break

    required_cols = ["Open time", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        found = ", ".join(str(col) for col in df.columns)
        raise ValueError(
            f"Missing required column(s) in CSV: {', '.join(missing)}. "
            f"Columns found: {found}"
        )

    df["Datetime"] = pd.to_datetime(df["Open time"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["Datetime"]).reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("CSV has no valid OHLCV rows after parsing.")
    return df


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["Datetime"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Datetime"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def _poc_for_box(box_rows: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    if not box_rows:
        return np.nan, np.nan
    high, low, close, vol = max(box_rows, key=lambda item: item[3])
    return (high + low + close) / 3.0, vol


def run_backtest(
    csv_file_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    config: StrategyConfig = StrategyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _filter_dates(_load_ohlcv(csv_file_path), start, end)
    if df.empty:
        raise ValueError("No rows found for the requested date range.")

    alpha = 2.0 / (config.smoothing_len + 1)

    bar_counter = 0
    cum_vol = 0.0
    past_green_box_volumes: list[float] = []
    past_red_box_volumes: list[float] = []
    smooth_up_eff = np.nan
    smooth_down_eff = np.nan
    max_high = float(df["High"].iloc[0])
    min_low = float(df["Low"].iloc[0])
    start_open = float(df["Open"].iloc[0])
    box_rows: list[tuple[float, float, float, float]] = []

    position = 0
    entry_price = np.nan
    pending_entry = 0
    capital = config.initial_capital

    trades: list[dict[str, object]] = []
    boxes: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    for i, row in df.iterrows():
        dt = row["Datetime"]
        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        vol = float(row["Volume"])

        if pending_entry and position == 0:
            position = pending_entry
            entry_price = open_p
            trades.append(
                {
                    "time": dt,
                    "event": "Buy Entry" if position == 1 else "Sell Entry",
                    "price": entry_price,
                    "capital": capital,
                }
            )
            pending_entry = 0

        if position == 1:
            sl = entry_price * config.long_sl
            tp = entry_price * config.long_tp
            if low_p <= sl:
                pnl_pct = ((sl - entry_price) / entry_price) * config.leverage
                capital += capital * pnl_pct
                trades.append({"time": dt, "event": "Exit Long (SL)", "price": sl, "capital": capital})
                position = 0
                entry_price = np.nan
            elif high_p >= tp:
                pnl_pct = ((tp - entry_price) / entry_price) * config.leverage
                capital += capital * pnl_pct
                trades.append({"time": dt, "event": "Exit Long (TP)", "price": tp, "capital": capital})
                position = 0
                entry_price = np.nan
        elif position == -1:
            sl = entry_price * config.short_sl
            tp = entry_price * config.short_tp
            if high_p >= sl:
                pnl_pct = ((entry_price - sl) / entry_price) * config.leverage
                capital += capital * pnl_pct
                trades.append({"time": dt, "event": "Exit Short (SL)", "price": sl, "capital": capital})
                position = 0
                entry_price = np.nan
            elif low_p <= tp:
                pnl_pct = ((entry_price - tp) / entry_price) * config.leverage
                capital += capital * pnl_pct
                trades.append({"time": dt, "event": "Exit Short (TP)", "price": tp, "capital": capital})
                position = 0
                entry_price = np.nan

        bar_counter += 1
        cum_vol += vol
        max_high = max(max_high, high_p)
        min_low = min(min_low, low_p)
        box_rows.append((high_p, low_p, close_p, vol))

        if bar_counter == 1:
            start_open = open_p

        if bar_counter >= config.box_size:
            total_box_height = max_high - min_low
            is_bullish_box = close_p >= start_open
            is_weekend = dt.dayofweek in (5, 6)
            allow_signal = (not config.filter_weekend) or (not is_weekend)
            expected_move = np.nan
            avg_vol = np.nan
            signal = 0

            if is_bullish_box:
                avg_vol = float(np.mean(past_green_box_volumes)) if past_green_box_volumes else np.nan
                vol_pass = (
                    (not config.enable_vol_imbalance)
                    or np.isnan(avg_vol)
                    or (cum_vol >= avg_vol * config.vol_ratio_threshold)
                )
                current_eff = total_box_height / (cum_vol if cum_vol > 0 else 1.0)

                if not np.isnan(smooth_up_eff):
                    expected_move = smooth_up_eff * cum_vol
                    compressed = (expected_move * config.sensitivity_factor) > total_box_height
                    if compressed and vol_pass and allow_signal and position == 0 and pending_entry == 0:
                        signal = -1

                smooth_up_eff = current_eff if np.isnan(smooth_up_eff) else (
                    alpha * current_eff + (1.0 - alpha) * smooth_up_eff
                )
                past_green_box_volumes.append(cum_vol)
                past_green_box_volumes = past_green_box_volumes[-8:]
            else:
                avg_vol = float(np.mean(past_red_box_volumes)) if past_red_box_volumes else np.nan
                vol_pass = (
                    (not config.enable_vol_imbalance)
                    or np.isnan(avg_vol)
                    or (cum_vol >= avg_vol * config.vol_ratio_threshold)
                )
                current_eff = total_box_height / (cum_vol if cum_vol > 0 else 1.0)

                if not np.isnan(smooth_down_eff):
                    expected_move = smooth_down_eff * cum_vol
                    compressed = (expected_move * config.sensitivity_factor) > total_box_height
                    if compressed and vol_pass and allow_signal and position == 0 and pending_entry == 0:
                        signal = 1

                smooth_down_eff = current_eff if np.isnan(smooth_down_eff) else (
                    alpha * current_eff + (1.0 - alpha) * smooth_down_eff
                )
                past_red_box_volumes.append(cum_vol)
                past_red_box_volumes = past_red_box_volumes[-2:]

            poc_price, poc_vol = _poc_for_box(box_rows)
            boxes.append(
                {
                    "time": dt,
                    "box": len(boxes) + 1,
                    "direction": "green" if is_bullish_box else "red",
                    "volume": cum_vol,
                    "avg_prior_volume": avg_vol,
                    "actual_height": total_box_height,
                    "expected_height": expected_move,
                    "poc_price": poc_price,
                    "poc_volume": poc_vol,
                    "signal": "Sell" if signal == -1 else "Buy" if signal == 1 else "",
                }
            )

            if signal:
                pending_entry = signal

            bar_counter = 0
            cum_vol = 0.0
            max_high = high_p
            min_low = low_p
            start_open = open_p
            box_rows = []

        equity_rows.append({"time": dt, "capital": capital, "position": position})

    return pd.DataFrame(trades), pd.DataFrame(equity_rows), pd.DataFrame(boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python backtest matched to the Pine fixed-box strategy.")
    default_csv = Path("/Users/yogeshkadam/Desktop/Projects/TradingBacktest /Eth /BYBIT_ETHUSDT_15m.csv")
    if not default_csv.exists():
        default_csv = Path("eth_15m_data_2017_to_2025.csv")
    parser.add_argument("csv", nargs="?", default=str(default_csv))
    parser.add_argument("--start", default=None, help="Optional inclusive start date, for example 2026-06-01")
    parser.add_argument("--end", default=None, help="Optional inclusive end date, for example 2026-06-30 23:59:59")
    parser.add_argument("--trades-out", default="matched_strategy_trades.csv")
    parser.add_argument("--equity-out", default="matched_strategy_equity.csv")
    parser.add_argument("--boxes-out", default="matched_strategy_boxes.csv")
    args, _unknown = parser.parse_known_args()

    trades, equity, boxes = run_backtest(args.csv, args.start, args.end)
    trades.to_csv(args.trades_out, index=False)
    equity.to_csv(args.equity_out, index=False)
    boxes.to_csv(args.boxes_out, index=False)

    final_capital = equity["capital"].iloc[-1]
    print("Backtest complete")
    print(f"Rows tested: {len(equity)}")
    print(f"Boxes closed: {len(boxes)}")
    print(f"Trade events: {len(trades)}")
    print(f"Initial capital: $10,000.00")
    print(f"Final capital: ${final_capital:,.2f}")
    print(f"Net return: {(final_capital / 10_000.0 - 1.0) * 100:.2f}%")
    print(f"Wrote: {args.trades_out}, {args.equity_out}, {args.boxes_out}")


if __name__ == "__main__":
    main()
