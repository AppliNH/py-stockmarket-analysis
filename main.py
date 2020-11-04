import finnhub
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

import analyze

now = datetime.now()
year_ago = now - timedelta(days=1*365)
FINNHUB_KEY = ""

def create_df_from_stock_market(symbol: str,verbose:bool=False):

    print(symbol)

    finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
    res = finnhub_client.stock_candles(
        symbol,
        'D',
        int(datetime.timestamp(year_ago)),
        int(datetime.timestamp(now)))
    
    if res["s"] != "ok":
        print(res)

    if verbose==True:
        print(res)

    df = pd.DataFrame(data=res)
    df.drop(columns="s", inplace=True)
  
    return df

def generate_final_df_from_symbols(symbols):
    dfs = []
    for symbol in symbols:
        df = create_df_from_stock_market(symbol=symbol)
        dfs.append(df.set_index("t"))
    
    dict_df = dict(zip(symbols, dfs))
    concat_df = pd.concat(dict_df, axis=1)
    return concat_df



def main():
    global FINNHUB_KEY
    load_dotenv()
    FINNHUB_KEY = os.getenv('FINNHUB_KEY')

    # Social media companies stocks dataset
    social_df = generate_final_df_from_symbols(symbols=["FB","SNAP","TWTR","PINS"])
    social_df.to_csv("social_medias_stock_df.csv")
    print("social_medias_stock_df.csv done")

    # Automotive companies stocks dataset
    auto_df = generate_final_df_from_symbols(symbols=["GM","BMWYY","VLKAF", "PUGOY"])
    auto_df = analyze.automobile_stocks_pre_processing(auto_df)
    auto_df.to_csv("automobile_stock_df.csv")
    print("automobile_stock_df.csv done")

    print(social_df)
    print("____________")
    print(auto_df)


if __name__ == "__main__":
    main()
