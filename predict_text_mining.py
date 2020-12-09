import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np
import mplfinance as mpf
from pandas.tseries.offsets import BDay
import requests
import time
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class StockNews:

    def __init__(self, ticker: str, title:str, date: datetime.datetime):
        self.ticker = ticker
        self.title = title
        self.date = date

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "title": self.title,
            "date": self.date
        }


def text_minining_sentiment(ticker: str, df: pd.DataFrame):

    load_dotenv()
    STOCKNEWSAPI_KEY = os.getenv('STOCKNEWSAPI_KEY')

    url ="https://stocknewsapi.com/api/v1?tickers="+ticker+"&items=50&token="+STOCKNEWSAPI_KEY

    r = requests.get(url)

    data = r.json()
    

    articles = data["data"]

    stock_news_list = []

    if len(articles) > 0:
        for article in articles:

            headline = article["title"] # News headline
            date_str = article["date"].split() # Date
            
            try:
                date = datetime.datetime.strptime(('-'.join(date_str[1:4])), '%d-%b-%Y')
            except:
                return
            if date <= datetime.datetime.strptime('03-12-2020', '%d-%m-%Y'):
                x = StockNews(ticker, headline, date)
                stock_news_list.append(x)
            

        vader = SentimentIntensityAnalyzer()

        parsed_and_scored_news = pd.DataFrame.from_records([news.to_dict() for news in stock_news_list], columns=["ticker", "title", "date"])


        scores = parsed_and_scored_news['title'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
        parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
        
        plot_score_df = parsed_and_scored_news[["date", "compound"]].copy()
        plot_score_df = plot_score_df.set_index("date")
        
        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        
        print(df["c"].loc[plot_score_df.index[-1] : plot_score_df.index[0] ])
        df["c"].loc[plot_score_df.index[-1] : plot_score_df.index[0] ].plot(ax=axes[0])

        axes[0].set_ylabel("Price ($)")
        axes[0].set_title(ticker +" closing price evolution" )

        plot_score_df_bis = plot_score_df[(plot_score_df != 0).all(1)].reset_index()
        plot_score_df_bis = plot_score_df_bis.rename(columns={"compound": "score"})

        plot_score_df_bis.plot(ax=axes[1], kind="scatter",x="date", y="score")
        

        axes[1].set_ylabel("Sentiment score")
        axes[1].set_title(ticker +" sentiment score evolution" )

        plt.show()
        

        # _____ Correlation between sentiment score and closing prices evolution
        merged_evolution_sentimentscore_df = pd.DataFrame()
        merged_evolution_sentimentscore_df["closes"] = df["c"]

        # Removing time from datetime index
        merged_evolution_sentimentscore_df.index = merged_evolution_sentimentscore_df.index.normalize()

        # Filtering closing prices rows from sentiment score available rows
        merged_evolution_sentimentscore_df = merged_evolution_sentimentscore_df[merged_evolution_sentimentscore_df.index.isin(plot_score_df.index)]

        # Merge and drop NaN values
        merged_evolution_sentimentscore_df = merged_evolution_sentimentscore_df.join(plot_score_df, how="outer")
        merged_evolution_sentimentscore_df = merged_evolution_sentimentscore_df.dropna()
        merged_evolution_sentimentscore_df = merged_evolution_sentimentscore_df[(merged_evolution_sentimentscore_df != 0).all(1)]

        merged_evolution_sentimentscore_df["closes"] = merged_evolution_sentimentscore_df["closes"].rolling(window=5).mean()
        merged_evolution_sentimentscore_df["compound"] = merged_evolution_sentimentscore_df["compound"].add(1).rolling(window=5).mean()
        merged_evolution_sentimentscore_df = merged_evolution_sentimentscore_df.dropna()


        # Column rename
        merged_evolution_sentimentscore_df = merged_evolution_sentimentscore_df.rename(columns={"compound": "sentiment score"})

        print(merged_evolution_sentimentscore_df)

        
        sns.heatmap(merged_evolution_sentimentscore_df.corr(), cmap="Reds", annot=True)
        plt.title(ticker +" : Correlation between closing price evolution")
        plt.show()
       
        
    
    # https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
    # https://programminghistorian.org/en/lessons/sentiment-analysis


def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))#.replace(hour=0, minute=0, second=0, microsecond=0)


if __name__ == "__main__":


    plt.style.use("seaborn")

    automobile_df = pd.read_csv("automobile_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
    social_df = pd.read_csv("social_medias_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
    prev=""

    # Unfortunately, this "for" loop randomly causes datetime conversion to fail.
    # Idk why

    # dfs = [automobile_df, social_df]
    # for df in dfs:
    #     for column,_ in df.columns:
    #         if prev != column:
    #             print(column)
    #             the_df = df[column].copy()
    #             text_minining_sentiment(column, the_df)
    #             prev = column


    # text_minining_sentiment("BMWYY", automobile_df["BMWYY"])
    # text_minining_sentiment("PUGOY", automobile_df["PUGOY"])
    
    # text_minining_sentiment("FB", social_df["FB"])
    text_minining_sentiment("SNAP", social_df["SNAP"])
    # text_minining_sentiment("PINS", social_df["PINS"])
    