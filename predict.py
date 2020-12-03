import finnhub
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import time
import os
from dotenv import load_dotenv


# ML packages
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def dec_tree_regr(df, x_future, X=None, y=None, tree_model=None, days: int=None, target_accuracy_score: float=None):
    # LinearRegression doesn't work well in this case, the accuracy score is too low..
    accuracy_score = None
    # Model creation
    if tree_model is None:
        tree_model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)

        # Model training
        tree_model.fit(X_train, y_train)

        tree_pred_test = tree_model.predict(X_test)
        # Evaluate accuracy
        accuracy_score = r2_score(y_test, tree_pred_test)
        # print("Accuracy score: "+str(accuracy_score))

    # Model predict
    tree_prediction = tree_model.predict(x_future)

    predictions = tree_prediction

    if X is not None and y is not None:
        # print("Iterating...")
        if (accuracy_score < target_accuracy_score) == False: # If we reach target_accuracy_score
            plt.figure(figsize=(16, 8))
            plt.xlabel('Days', fontsize=18)
            plt.ylabel('Close Price USD ($)', fontsize=18)
            plt.title('Model with '+str(accuracy_score*100)+'%\ accuracy')
            valid = df[X.shape[0]:]
            valid['Prediction'] = predictions

            plt.plot(df['c'])
            plt.plot(valid[['c', 'Prediction']])

            plt.legend(['Train', 'Val', 'Prediction'], loc='lower right')
            plt.show()
    else:
        # Visualize the data
        plt.figure(figsize=(16, 8))
        plt.xlabel('Days', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.title(str(days)+' days predict')
        pred_df = pd.DataFrame(data=predictions, columns=["Prediction"])
        pred_df["c"] = np.nan
        valid = df.drop(['Prediction'], 1)
        valid["Prediction"] = np.nan
        valid = valid.append(pred_df, ignore_index=True)

        plt.plot(valid[['c', 'Prediction']], scaley=True)
        plt.legend(['Known','Prediction'], loc='lower right')
        plt.show()

    return tree_model, accuracy_score


def predict_stock_market(df_to_analyze ,nb_future_days: int, desired_accuracy: float):

    now = df_to_analyze.index[len(df_to_analyze.index)-1] # last date

    
    # Stock candles

    closes = df_to_analyze[["c"]].copy()

    closes["Prediction"] = closes[["c"]].shift(-25)

    X = np.array(closes.drop(['Prediction'], 1))[:-25]
    y = np.array(closes['Prediction'])[:-25]

    x_future = closes.drop(['Prediction'], 1)[:-25]
    x_future = x_future.tail(25)
    x_future = np.array(x_future)

    # Train model on guessing for 25 days (but data is already known)
    accuracy_score = 0
    while accuracy_score < desired_accuracy:
        model, accuracy_score = dec_tree_regr(closes, x_future, X, y, tree_model=None, target_accuracy_score=desired_accuracy)     

    x_future = closes.drop(['Prediction'], 1)
    x_future = x_future.tail(nb_future_days) # Predict true future for five days
    x_future = np.array(x_future)

    dec_tree_regr(closes, x_future, tree_model=model, days=nb_future_days)


def dateparse (time_in_secs):    
    return datetime.fromtimestamp(float(time_in_secs))#.replace(hour=0, minute=0, second=0, microsecond=0)


def main():
    plt.style.use("seaborn")

    automobile_df = pd.read_csv("automobile_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)
    social_df = pd.read_csv("social_medias_stock_df.csv", header=[0,1], index_col=0, parse_dates=True,date_parser=dateparse)


    load_dotenv()
    FINNHUB_KEY = os.getenv('FINNHUB_KEY')
    if FINNHUB_KEY is None:
        print("You haven't provided any FinnHub API KEY.")
        print("- Go to https://finnhub.io/")
        print("- Get yourself a key")
        print("- Add an .env file")
        print("- Write FINNHUB_KEY=YOUR_API_KEY in there")
        print("Then you should be good to go :)")
        return

    prev=""
    
    dfs = [automobile_df, social_df]
    for df in dfs:
        for column,_ in df.columns:
            print(column)
            if prev != column:
                the_df = df[column].copy()
                predict_stock_market(the_df, nb_future_days=2, desired_accuracy=0.80)
                prev = column

if __name__ == "__main__":
    main()
