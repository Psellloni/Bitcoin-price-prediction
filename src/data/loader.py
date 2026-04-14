from typing import Dict, List, Tuple, Any
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
import numpy as np

pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_columns = None

# parsing all of the currency pairs & crypto coins

def get_data(
      ticker_names: Dict[str, str]
    , start_date: str = str(date.today() - relativedelta(years=1))
    , end_date: str = str(date.today())
    , interval : str = "1wk"
    , price_type:  List[str]|str = 'Close'
    , ignore_info: bool = False
):
    """
    собираем данные для дальнейшей работы
    
    :param ticket_names: название тикера и его аббревиатура
    :param start_date: начальная дата
    :param end_date: последняя дата
    :param interval: интервал наблюдений, Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    :param price_type: тип цены (Open / High / Low / Close / Adj Close)
    """

    def print_info(days: int = 7, all_good: bool = False) -> None:
        if all_good:
            print(f"Данные успешно загружены\n")
            print('-------------------------------------')            
        else:
            print(f"Временные рамки заменены. Данные собраны с {str(date.today() - relativedelta(days=days))} до \
                                                                            {str(date.today())}")
            print('-------------------------------------')

    if interval in ('2m', '5m', '15m', '30m', '90m'):
        data = yf.download(list(ticker_names), start=date.today() - relativedelta(days=59), end=date.today(), interval=interval)
        data = data.loc[:, data.columns.get_level_values(0).isin(price_type)]
        data.columns = [ticker_names[ticker_name] + '_' + price_type for price_type, ticker_name in data.columns]
        
        if not ignore_info:
            print_info(days=60)

    elif interval == '1m':
        data = yf.download(list(ticker_names), start=date.today() - relativedelta(days=7), end=date.today(), interval=interval)
        data = data.loc[:, data.columns.get_level_values(0).isin(price_type)]
        data.columns = [ticker_names[ticker_name] + '_' + price_type for price_type, ticker_name in data.columns]
        
        if not ignore_info:
            print_info(days=7)

    elif interval in ('60m', '1h') and (pd.to_datetime(end_date) - 
                                        pd.to_datetime(start_date)).days > 730:
        data = yf.download(list(ticker_names), start=date.today() - relativedelta(days=729), end=date.today(), interval=interval)
        data = data.loc[:, data.columns.get_level_values(0).isin(price_type)]
        data.columns = [ticker_names[ticker_name] + '_' + price_type for price_type, ticker_name in data.columns]
        
        if not ignore_info:
            print_info(days=730)

    else:
        data = yf.download(list(ticker_names), start=start_date, end=end_date, interval=interval)
        data = data.loc[:, data.columns.get_level_values(0).isin(price_type)]
        data.columns = [ticker_names[ticker_name] + '_' + price_type for price_type, ticker_name in data.columns]
        
        if not ignore_info:
            print_info(all_good=True)

    return data