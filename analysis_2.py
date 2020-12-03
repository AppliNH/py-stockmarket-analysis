import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np
import mplfinance as mpf
from pandas.tseries.offsets import BDay


def spotBullishEngulfinPattern(df: pd.DataFrame, stock_name: str):

    BEp_df = pd.DataFrame()

    for i in range(2,df.shape[0]):
        current = df.iloc[i,:]
        prev = df.iloc[i-1,:]
        # prev_2 = df.iloc[i-2,:]
        realbody = abs(current['Open'] - current['Close'])
        candle_range = current['High'] - current['Low']
        idx = df.index[i]
        BEp_df.loc[idx,'Bullish engulfing'] = current['High'] > prev['High'] and current['Low'] < prev['Low'] and realbody >= 0.8 * candle_range and current['Close'] > current['Open']
    
    spot_idx = BEp_df.index[BEp_df["Bullish engulfing"] == True].tolist()
    print(spot_idx)

    for idx in spot_idx:
        print(idx)

        target = idx +  BDay(5)
        perf_str = ""
        if target in df.Close.index:
            perf = (df.Close.loc[target] / df.Close.loc[idx]) * 100
            perf_str = "Price evolution 5 days after pattern appearance : "+str(perf)+"%"
            print("perf is : "+str(perf) +"%")

        # Display BE pattern on close prices evolution graph
        df.Close.loc[idx - datetime.timedelta(days=20):idx + datetime.timedelta(days=20)].plot(figsize=(13,8))
        plt.figtext(.5, .95, stock_name + " : Bullish engulfing pattern location.",ha='center')
        if perf_str != "":
            plt.figtext(.5,.9,perf_str,ha='center')
        plt.ylabel("Price ($)")
        plt.axvline(x=idx, color="red")
        plt.show()

        # Display BE pattern
        df_spotBEp = df.loc[idx - datetime.timedelta(days=5):idx + datetime.timedelta(days=10)]
        mpf.plot(df_spotBEp, ylabel="Price ($)", type="candle", mav=4,title=stock_name + " : Bullish engulfing pattern of "+str(idx.date()))
        plt.show()

        
        

        





    

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))#.replace(hour=0, minute=0, second=0, microsecond=0)

plt.style.use("seaborn")

automobile_df = pd.read_csv("automobile_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
social_df = pd.read_csv("social_medias_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)

# _____________________ Spot bullish engulfing pattern _____________________________

print("_____________________ Spot bullish engulfing pattern _____________________________")
prev=""

dfs = [automobile_df, social_df]
for df in dfs:
    for column,_ in df.columns:
        if prev != column:
            print(column)
            the_df = df[column].copy()
            the_df = the_df.rename(columns={"c": "Close", "h": "High", "l": "Low", "v":"Volume", "o":"Open"})
            spotBullishEngulfinPattern(the_df,column)
            prev = column
