import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))#.replace(hour=0, minute=0, second=0, microsecond=0)

plt.style.use("seaborn")

automobile_df = pd.read_csv("automobile_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
social_df = pd.read_csv("social_medias_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)


automobile_closes_df = automobile_df.loc[:, (slice(None), 'c')]
social_closes_df = social_df.loc[:, (slice(None), 'c')]

automobile_closes_df.columns = automobile_closes_df.columns.droplevel(level=1)
social_closes_df.columns = social_closes_df.columns.droplevel(level=1)



# First display : Evolution of closing prices

automobile_closes_df.plot()
plt.title("Evolution over a year of closing prices for companies of automotive industry")
plt.ylabel("Price ($)")



social_closes_df.plot()
plt.title("Evolution over a year of closing prices for companies of social media industry")
plt.ylabel("Price ($)")


# Seconding display : Performance evolution of closing prices

automotive_closes_normalize_from_first = automobile_closes_df.div(automobile_closes_df.iloc[0]).mul(100).copy()

automotive_closes_normalize_from_first.plot()
plt.title("Performance evolution of stocks for automotive industry")
plt.ylabel("Performance (%)")


social_closes_normalize_from_first = social_closes_df.div(social_closes_df.iloc[0]).mul(100).copy()

social_closes_normalize_from_first.plot()
plt.title("Performance evolution of stocks for social medias industry")
plt.ylabel("Performance (%)")


plt.show()


# Stock performance : distribution of daily returns
print("_______________Stock performance :  Daily returns_________________________________")

ret_autos = automobile_closes_df.pct_change().dropna().copy()
ret_autos.plot(kind="hist", figsize=(12,8), bins=100, subplots=True, sharey=True, title="Stock performance : distribution of daily returns for companies of automotive industry")
plt.xlabel("Daily returns percentage")

plt.show()

ret_social = social_closes_df.pct_change().dropna().copy()
ret_social.plot(kind="hist", figsize=(12,8), bins=100, subplots=True, sharey=True, title="Stock performance : distribution of daily returns for companies of social medias industry")
plt.xlabel("Daily returns percentage")

plt.show()

ret_autos_mean = ret_autos.mean()
ret_social_mean = ret_social.mean()

ret_autos_std = ret_autos.std()
ret_social_std = ret_social.std()

print("Means : ")
print(ret_autos_mean)
print(ret_social_mean)

print("Standard deviation : => The higher, the more losses or profits !")
print(ret_autos_std)
print(ret_social_std)

ret_autos_mean.plot(kind="bar")
plt.title("Daily returns means for companies of automotive industry")
plt.ylabel("Daily returns percentage")



ret_autos_std.plot(kind="bar")
plt.title("Daily returns standard deviation for companies of automotive industry")
plt.ylabel("STD Value")




ret_social_mean.plot(kind="bar")
plt.title("Daily returns means for companies of social medias industry")
plt.ylabel("Daily returns percentage")




ret_social_std.plot(kind="bar")
plt.title("Daily returns standard deviation for companies of social medias industry")
plt.ylabel("STD Value")



plt.show()

# ___________________Return and risk_____________________

print("____________Return and risk_______________")

ret_autos_desc = ret_autos.describe().T.copy() # .T here allows to transpose the statistics, so count,mean,std and else are columns
ret_social_desc = ret_social.describe().T.copy()

ret_autos_mean_and_std = ret_autos_desc.loc[:,["mean", "std"]] # Only keeping mean and std
ret_social_mean_and_std = ret_social_desc.loc[:,["mean", "std"]] # Only keeping mean and std

# Annualizing mean and std
# On average, a calendar has 252 trading days

ret_autos_mean_and_std["mean"] = ret_autos_mean_and_std["mean"]*252
ret_autos_mean_and_std["std"] = ret_autos_mean_and_std["std"]* np.sqrt(252)

ret_social_mean_and_std["mean"] = ret_social_mean_and_std["mean"]*252
ret_social_mean_and_std["std"] = ret_social_mean_and_std["std"]* np.sqrt(252)

# Rendering the graph for both industries

ax_autos = ret_autos_mean_and_std.plot.scatter(x= "std", y="mean", s=50, fontsize=15, color="r", label="Companies from automotive industry")

for i in ret_autos_mean_and_std.index:
    ax_autos.annotate(i, xy=(ret_autos_mean_and_std.loc[i, "std"]+0.002, ret_autos_mean_and_std.loc[i, "mean"]+0.002), size=13)

ret_social_mean_and_std.plot.scatter(x= "std", y="mean", s=50, fontsize=15, figsize=(12,8),color="g", ax=ax_autos, label="Companies from social medias industry")

for i in ret_social_mean_and_std.index:
    plt.annotate(i, xy=(ret_social_mean_and_std.loc[i, "std"]+0.002, ret_social_mean_and_std.loc[i, "mean"]+0.002), size=13)


plt.xlabel("Annual Risk (std)")
plt.ylabel("Annual Return")
plt.title("Risk and Return for stocks of companies from automotive and social medias industry")

plt.show()

## _________________ Covariance and Correlation of stocks ___________________________

print("_________________ Covariance and Correlation of stocks ___________________________")

# Correlation
auto_corr = automobile_closes_df.corr().copy()
social_corr = social_closes_df.corr().copy()


print(auto_corr)
print(social_corr)

print(auto_corr.mean())
print(social_corr.mean())
# Automotive stocks are more subject to correlations than social media stocks !

# Heatmaps of correlations

plt.figure(figsize=(13,8))
plt.title("Correlations for stocks of automotive industry")
sns.heatmap(auto_corr, cmap="Reds", annot=True,fmt=".2%")

plt.figure(figsize=(13,8))
plt.title("Correlations for stocks of social media industry")
sns.heatmap(social_corr, cmap="Reds", annot=True,fmt=".2%")

plt.show()

## _________________ Rolling statistics and simple moving averages ___________________________

print("_________________ Rolling statistics and simple moving averages ___________________________")
# Aggreate mean for the previous 10 days => Window

print(automobile_closes_df.rolling(window=10,min_periods=10).mean())
print(social_closes_df.rolling(window=10, min_periods=10).mean())


## _________________Momentum trading strategies___________________________

print("_________________Momentum trading strategies___________________________")

# The shorter SMA (10) captures the most recent trends (momentum)
VLKAF_auto_SMA10 = automobile_closes_df.VLKAF.rolling(window=10).mean()
FB_social_SMA10 = social_closes_df.FB.rolling(window=10).mean()

# The longer SMA captures a more general trend
VLKAF_auto_SMA50 = automobile_closes_df.VLKAF.rolling(window=50).mean()
FB_social_SMA50 = social_closes_df.FB.rolling(window=50).mean()

# Traders invest when the shorter term SMA is above the longer term SMA

FB_social_div_10by50 = FB_social_SMA10.div(FB_social_SMA50).sub(1)
VLKAF_auto_div_10by50 = VLKAF_auto_SMA10.div(VLKAF_auto_SMA50).sub(1)


plt.figure(figsize=(13,8))
VLKAF_auto_SMA50.plot()
VLKAF_auto_SMA10.plot()
plt.ylabel("Stock value ($)")
plt.legend(["SMA Window 50", "SMA Window 10"])
plt.title("Simple Moving Averages for VLKAF stock with short and longer windows")



plt.figure(figsize=(13,8))
FB_social_SMA50.plot()
FB_social_SMA10.plot()
plt.ylabel("Stock value ($)")
plt.legend(["SMA Window 50", "SMA Window 10"])
plt.title("Simple Moving Averages for FB stock with short and longer windows")


plt.figure(figsize=(13,8))
FB_social_div_10by50.plot()
VLKAF_auto_div_10by50.plot()
plt.ylabel("Invest rate")
plt.legend(["FB potential invest rate", "VLKAF potential invest rate"])
plt.figtext(.5,0.95,"Comparing potential investment in FB and VLKAF stocks over the year",fontsize=12, ha='center')
plt.figtext(.5,.9,'According to the Momentum trading strategy, traders are less likely to invest when the invest rate goes beyond 0',fontsize=10,ha='center')


plt.show()


# With rolling risk and daily returns
FB_ret = pd.DataFrame()
FB_ret["close"] =  social_closes_df.FB.pct_change().dropna()
FB_ret["return"] = FB_ret.rolling(window=10).mean()*252
FB_ret["risk"] = FB_ret.close.rolling(window=10).std()*np.sqrt(252)
FB_ret.drop(columns="close", inplace=True)


VLKAF_ret = pd.DataFrame()
VLKAF_ret["close"] =  automobile_closes_df.VLKAF.pct_change().dropna()
VLKAF_ret["return"] = VLKAF_ret.rolling(window=10).mean()*252
VLKAF_ret["risk"] = VLKAF_ret.close.rolling(window=10).std()*np.sqrt(252)
VLKAF_ret.drop(columns="close", inplace=True)

FB_ret.plot()
plt.title("FB stocks rolling risk and return")
plt.ylabel("Rate (%)")
plt.show()


VLKAF_ret.plot()
plt.title("VLKAF stocks rolling risk and return")
plt.ylabel("Rate (%)")
plt.show()


# Allows to see wether an investment with high return is risky or not
ax = FB_ret.plot.scatter(x="risk", y="return", color="b")
VLKAF_ret.plot.scatter(ax=ax, x="risk", y="return", color="r")
plt.title("Comparing FB and VLKAF stocks on daily returns and risk rates")
plt.ylabel("Return rate (%)")
plt.xlabel("Risk rate (%)")
plt.legend(["FB", "VLKAF"])
plt.show()


## ______________________DRAFT________________________


# The minimal closes price over the last year happened all within the same week

# print("MINIMAL CLOSE PRICES FOR AUTOMOTIVE INDUSTRY")
# print(automobile_closes_df.min())
# print("WITH DATES")
# print(automobile_closes_df.idxmin())

# This is also quite the same for maximal closes price, happened in the end of 2019 / beggining of 2020

# print(automobile_closes_df.max())
# print(automobile_closes_df.idxmax())

# print("________________________________________________")

# The minimal closes price over the last year happened all within the same week

# print("MINIMAL CLOSE PRICES FOR SOCIAL MEDIAS INDUSTRY")
# print(social_closes_df.min())
# print("WITH DATES")
# print(social_closes_df.idxmin())

# For maximal prices, it's a bit more complicated to evaluate
# print(social_closes_df.max())
# print(social_closes_df.idxmax())



# Social medias variations
# social_closes_df.diff(periods=1).plot(subplots=True, sharey=True)
# plt.show()

# print("__________NORMALIZING FROM FIRST VALUE______________________________________")

# Calculation pctg change
# for i in range(4):
#     print(social_closes_df.iloc[:,i].div(social_closes_df.iloc[0,i]))

