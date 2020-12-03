import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import seaborn as sns
import numpy as np
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def compute(df: pd.DataFrame):

    data = df.filter(["c"])
    dataset = data.values
    training_data_length = math.ceil( len(dataset) *.8) 

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_length  , : ]

    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    


    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)


    #Test data set
    test_data = scaled_data[training_data_length - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_length : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    #Convert x_test to a numpy array 
    x_test = np.array(x_test)

    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling


    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print(rmse)

    #Plot/Create the data for the graph
    train = data[:training_data_length]
    valid = data[training_data_length:]
    valid['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['c'])
    plt.plot(valid[['c', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()



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
            compute(the_df)
            prev = column