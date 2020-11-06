import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

automobile_df = pd.read_csv("automobile_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
#print(automobile_df)

social_df = pd.read_csv("social_medias_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
#print(social_df)


automobile_closes_df = automobile_df.loc[:, (slice(None), 'c')]
social_closes_df = social_df.loc[:, (slice(None), 'c')]

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
# print(social_closes_df.max())
# print(social_closes_df.idxmax())

print(social_closes_df.groupby("t").corr())



# automobile_closes_df.plot()
# plt.show()