from __future__ import annotations

from typing import Dict, Iterable, List, Union
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import yfinance as yf

pd.options.display.float_format = "{:.2f}".format
pd.options.display.max_columns = None


PriceType = Union[str, List[str]]

# Ограничения Yahoo Finance по максимальной глубине истории для некоторых интервалов
INTERVAL_LIMITS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "90m": 60,
    "60m": 730,
    "1d": 730
}


def _normalize_price_types(price_type: PriceType) -> List[str]:
    if isinstance(price_type, str):
        return [price_type]
    return list(price_type)


def _resolve_dates(
    start_date: str,
    end_date: str,
    interval: str,
) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    """
    Возвращает скорректированные start/end даты с учётом ограничений Yahoo Finance.

    Returns:
        start_ts: итоговая дата начала
        end_ts: итоговая дата конца
        dates_adjusted: были ли даты автоматически скорректированы
    """
    print(interval)
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)

    if requested_start >= requested_end:
        raise ValueError("start_date must be earlier than end_date")

    max_days = INTERVAL_LIMITS.get(interval)

    if max_days is None:
        return requested_start, requested_end, False

    final_start = pd.Timestamp(date.today() - relativedelta(days=max_days-1))
    final_end = pd.Timestamp(date.today())

    print(final_start, final_end)

    return final_start, final_end


def _flatten_columns(
    data: pd.DataFrame,
    ticker_names: Dict[str, str],
    price_types: List[str],
) -> pd.DataFrame:
    """
    Преобразует MultiIndex-колонки вида:
    ('Close', 'BTC-USD') -> 'BTC_Close'
    """
    if data.empty:
        return data

    if not isinstance(data.columns, pd.MultiIndex):
        return data

    data = data.loc[:, data.columns.get_level_values(0).isin(price_types)].copy()
    data.columns = [
        f"{ticker_names[ticker]}_{field}"
        for field, ticker in data.columns
    ]
    return data


def get_data(
    ticker_names: Dict[str, str],
    start_date: str = str(date.today() - relativedelta(years=1)),
    end_date: str = str(date.today()),
    interval: str = "1wk",
    price_type: PriceType = "Close",
    ignore_info: bool = False,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Загружает котировки из Yahoo Finance.

    Parameters
    ----------
    ticker_names : Dict[str, str]
        Словарь вида {'BTC-USD': 'BTC', 'ETH-USD': 'ETH'}.
    start_date : str
        Дата начала в формате YYYY-MM-DD.
    end_date : str
        Дата конца в формате YYYY-MM-DD.
    interval : str
        Интервал данных.
        Допустимые значения:
        [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    price_type : str | List[str]
        Типы цен: Open / High / Low / Close / Adj Close / Volume
    ignore_info : bool
        Если False, печатает информацию о фактическом диапазоне дат.
    auto_adjust : bool
        Передать в yfinance auto_adjust.
    progress : bool
        Показывать ли progress bar yfinance.

    Returns
    -------
    pd.DataFrame
        DataFrame с плоскими колонками, например:
        BTC_Close, ETH_Close, BTC_Volume
    """
    if not ticker_names:
        raise ValueError("ticker_names cannot be empty")

    price_types = _normalize_price_types(price_type)
    final_start, final_end = _resolve_dates(start_date, end_date, interval)

    print(ticker_names, start_date, final_start)
    print(ticker_names, end_date, final_end)

    data = yf.download(
        tickers=list(ticker_names.keys()),
        start=final_start.strftime("%Y-%m-%d"),
        end=final_end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="column",
    )

    data = _flatten_columns(data, ticker_names=ticker_names, price_types=price_types)

    print(
        f"[INFO] Data successfully loaded: "
        f"{final_start.date()} to {final_end.date()}."
    )
    print("-" * 50)

    return data.reset_index()

def load_multiple_intervals(
    ticker_names: Dict[str, str],
    intervals: Iterable[str],
    start_date: str,
    end_date: str,
    price_type: PriceType = ("High", "Low", "Close", "Volume"),
    fill_method: str | None = "bfill",
    dropna: bool = True,
    ignore_info: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Загружает данные сразу для нескольких интервалов.
    """
    result: Dict[str, pd.DataFrame] = {}

    for interval in intervals:
        df = get_data(
            ticker_names=ticker_names,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            price_type=price_type,
            ignore_info=ignore_info,
        )

        if fill_method == "bfill":
            df = df.bfill()
        elif fill_method == "ffill":
            df = df.ffill()

        if dropna:
            df = df.dropna()

        result[interval] = df

    return result