import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np
import mplfinance as mpf
from pandas.tseries.offsets import BDay
import requests
# from textblob import TextBlob
# import re
# import nltk
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StockNews:

    def __init__(self, symbol: str, title:str, text: str, date: datetime.datetime):
        self.symbol = symbol
        self.title = title
        self.text = text
        self.date = date

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))#.replace(hour=0, minute=0, second=0, microsecond=0)

plt.style.use("seaborn")

automobile_df = pd.read_csv("automobile_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
social_df = pd.read_csv("social_medias_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)

prev=""

dfs = [automobile_df, social_df]
for df in dfs:
    for column,_ in df.columns:
        if prev != column:
            print(column)
            the_df = df[column].copy()
            prev = column


if __name__=="__main__":
    # url = ('http://newsapi.org/v2/top-headlines?'
    #    'q=GM&'
    #    'from=2020-12-02&'
    #    'language=en&'
    #    'sortBy=popularity&'
    #    'apiKey=a836223262b74409aed69044a145b1a3')

    # https://fmpcloud.io/documentation#stockNews
    # urlFB = "https://fmpcloud.io/api/v3/stock_news?tickers=FB&limit=1000&apikey=627c46d43f7489e510dfeffa51ea9c77"
    # urlPINS = "https://fmpcloud.io/api/v3/stock_news?tickers=PINS&limit=1000&apikey=627c46d43f7489e510dfeffa51ea9c77"
    # urlSNAP = "https://fmpcloud.io/api/v3/stock_news?tickers=SNAP&limit=1000&apikey=627c46d43f7489e510dfeffa51ea9c77"
    # urlTWTR = "https://fmpcloud.io/api/v3/stock_news?tickers=TWTR&limit=1000&apikey=627c46d43f7489e510dfeffa51ea9c77"

    # urls = [urlFB, urlPINS, urlSNAP, urlTWTR]

    # stock_news_list = []

    # for url in urls:
    #     response = requests.get(url)
    #     res_json = response.json()
    #     for item in res_json:
    #         x = StockNews(item["symbol"], item["title"], item["text"], datetime.datetime.strptime(item["publishedDate"],'%Y-%m-%d %H:%M:%S' ))
    #         stock_news_list.append(x)
            
    # for news in stock_news_list:
    #     print(news.date)

    