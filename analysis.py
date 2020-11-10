import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

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
plt.show()


social_closes_df.plot()
plt.title("Evolution over a year of closing prices for companies of social media industry")
plt.ylabel("Price ($)")
plt.show()

# Seconding display : Performance evolution of closing prices

automotive_closes_normalize_from_first = automobile_closes_df.div(automobile_closes_df.iloc[0]).mul(100).copy()

automotive_closes_normalize_from_first.plot()
plt.title("Performance evolution of stocks for automotive industry")
plt.ylabel("Performance (%)")
plt.show()

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
plt.show()

ret_autos_std.plot(kind="bar")
plt.title("Daily returns standard deviation for companies of automotive industry")
plt.ylabel("STD Value")
plt.show()

ret_social_mean.plot(kind="bar")
plt.title("Daily returns means for companies of social medias industry")
plt.ylabel("Daily returns percentage")
plt.show()

ret_social_std.plot(kind="bar")
plt.title("Daily returns standard deviation for companies of social medias industry")
plt.ylabel("STD Value")
plt.show()



## ______________________DRAFT________________________


# The minimal closes price over the last year happened all within the same week
print("MINIMAL CLOSE PRICES FOR AUTOMOTIVE INDUSTRY")
print(automobile_closes_df.min())
print("WITH DATES")
print(automobile_closes_df.idxmin())

# This is also quite the same for maximal closes price, happened in the end of 2019 / beggining of 2020
# print(automobile_closes_df.max())
# print(automobile_closes_df.idxmax())

print("________________________________________________")

# The minimal closes price over the last year happened all within the same week
print("MINIMAL CLOSE PRICES FOR SOCIAL MEDIAS INDUSTRY")
print(social_closes_df.min())
print("WITH DATES")
print(social_closes_df.idxmin())

# For maximal prices, it's a bit more complicated to evaluate
print(social_closes_df.max())
print(social_closes_df.idxmax())



# Social medias variations
# social_closes_df.diff(periods=1).plot(subplots=True, sharey=True)
# plt.show()

print("__________NORMALIZING FROM FIRST VALUE______________________________________")

# Calculation pctg change
# for i in range(4):
#     print(social_closes_df.iloc[:,i].div(social_closes_df.iloc[0,i]))

