# Importing libraries
import streamlit as st
import matplotlib.dates as mdates
import datetime
from pandas_datareader import data as pdr
import yfinance as yf #preffered over line above.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras import datasets, layers,models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation,SimpleRNN
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
tf.random.set_seed(7)

st.title('Stock Prediction using Deep Learning')
st.subheader('Select the method of input:')
# Adding radio buttons for the user to choose between Uploading csv and getting stock data from the net 
option = st.radio('Radio', ["Upload the data (.csv format)","Get data from the net"])
# creating a side bar
st.sidebar.title("Predicting FAANG Stock Prices")
st.sidebar.subheader("*Transforming stock market forecasting with AI*")
# Adding an image from Unsplash to the side bar
st.sidebar.image("https://assets.cmcmarkets.com/images/FAANG--1200px_extraExtra.webp", width=None)
# class for the deep learning modelsf
class stock_predict_DL:

    def __init__(self, comp_df):
        # Reserved method in Python classes (Constructor)

        # Handle MultiIndex columns and extract the correct level
        if isinstance(comp_df.columns, pd.MultiIndex):
            comp_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in comp_df.columns]

        # Map the renamed columns to the expected names
        column_mapping = {
            'Date_': 'Date',
            'Open_ABBV': 'Open',
        }
        comp_df.rename(columns=column_mapping, inplace=True)

        # Check for required columns
        required_columns = ['Open', 'Date']
        missing_columns = [col for col in required_columns if col not in comp_df.columns]
        if missing_columns:
            st.error(f"The dataset is missing the following required columns: {', '.join(missing_columns)}.")
            st.stop()

        # Convert 'Open' to numeric, coercing errors
        comp_df['Open'] = pd.to_numeric(comp_df['Open'], errors='coerce')
        comp_df = comp_df.dropna(subset=['Open'])

        # Validate 'Open' again
        if not pd.api.types.is_numeric_dtype(comp_df['Open']):
            st.error("The 'Open' column could not be converted to numeric. Please check the dataset.")
            st.stop()

        # Ensure 'Date' is valid and parseable
        try:
            comp_df['Date'] = pd.to_datetime(comp_df['Date'], errors='coerce')
            if comp_df['Date'].isnull().any():
                st.error("The 'Date' column contains invalid or missing datetime values. Please fix the dataset.")
                st.stop()
        except Exception as e:
            st.error(f"Error parsing 'Date' column: {e}")
            st.stop()

        # Filter data for 'Open'
        data = comp_df.filter(['Open'])
        dataset = data.values

        # User input for training percentage
        st.subheader('How much percent of the data needs to be allocated for training?')
        st.text('Default is set to 90')
        perc_train = st.number_input('', step=1, min_value=1, value=90)
        training_data_len = int(np.ceil(len(dataset) * (perc_train / 100)))

        # Scale data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(dataset)

        # Time window length
        k = 60  # Default time window length
        if len(dataset) <= k:
            st.error(f"Insufficient data. The dataset must have more than {k} rows.")
            st.stop()

        # Create training data
        train_data = scaled_data[:training_data_len, :]
        self.X_train, self.y_train = [], []
        for i in range(k, len(train_data)):
            self.X_train.append(train_data[i-k:i, 0])
            self.y_train.append(train_data[i, 0])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        # Create testing data
        test_data = scaled_data[training_data_len - k:, :]
        self.X_test, self.y_test = [], dataset[training_data_len:]
        for i in range(k, len(test_data)):
            self.X_test.append(test_data[i-k:i, 0])
        self.X_test = np.array(self.X_test)

        # Store test dates
        self.testd = comp_df['Date'].values[training_data_len:]


    def LSTM_model(self):

        st.title("Long Short-Term Memory (LSTM)")

        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(Xtrain, self.y_train, batch_size=1, epochs=1)

        # Get the model's predicted price values
        predictions = model.predict(Xtest)
        predictions = self.scaler.inverse_transform(predictions)

        # Metrics and Plotting
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, predictions))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, predictions))

        plt.plot(predictions, label="Predicted")
        plt.plot(self.y_test, label="Observed")
        plt.legend()

        # Adjust the x-axis for better date handling
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts ticks
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates
        plt.gcf().autofmt_xdate()  # Automatically rotates dates for better readability

        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.title("LSTM")
        st.pyplot(plt)

    def autoen_model(self):
        st.title("Autoencoder")
    
        # Number of encoding dimensions
        encoding_dim = 32
        input_dim = self.X_train.shape[1]

        # Building the Autoencoder
        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(1e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = Dense(1, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)

        # Compile and train the model
        nb_epoch = 10
        b_size = 32
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(self.X_train, self.y_train, epochs=nb_epoch, batch_size=b_size, shuffle=True)
    
        # Make predictions
        predictions = autoencoder.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(predictions)

        # Metrics and Plotting
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))

        # Plot the data
        plt.plot(y_pred, label="Predicted")
        plt.plot(self.y_test, label="Observed")
        plt.legend()

        # Adjust x-axis for better date handling
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjusts ticks
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates
        plt.gcf().autofmt_xdate()  # Automatically rotates dates for better readability

        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.title("AUTOENCODER")
        st.pyplot(plt)

    def Mlp_model(self):

        st.title("Multilayer perceptron (MLP)")
        # We are using MLPRegressor as the problem at hand is a regression problem
        regr = MLPRegressor(hidden_layer_sizes = 100, alpha = 0.01,solver = 'lbfgs',shuffle=True)
        regr.fit(self.X_train, self.y_train)
        # predicting the price
        y_pred = regr.predict(self.X_test)
        y_pred = y_pred.reshape(len(y_pred),1)
        y_pred = self.scaler.inverse_transform(y_pred)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))

        # Plotting
        plt.plot(y_pred, label="Predicted")
        plt.plot(self.y_test, label="Observed")
        plt.legend()

        # Adjust the x-axis for better date handling
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.title("MLP")
        st.pyplot(plt)

    def basic_ann_model(self):

        st.title("Basic Artificial Neural Network (ANN)")
        classifier = Sequential()
        classifier.add(Dense(units = 128, activation = 'relu', input_dim = self.X_train.shape[1]))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 64))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 1))
        # We are adding dropout to reduce overfitting
        # adam is one of the best optimzier for DL as it uses stochastic gradient method
        # Mean Square Error (MSE) is the most commonly used regression loss function.
        # MSE is the sum of squared distances between our target variable and predicted values.
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
        classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 10)
        # Predicting the prices
        prediction = classifier.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))


        # Plotting
        plt.plot(y_pred, label="Predicted")
        plt.plot(self.y_test, label="Observed")
        plt.legend()

        # Adjust the x-axis for better date handling
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.title("Basic ANN")
        st.pyplot(plt)



    def rnn_model(self):

        st.title("Recurrent neural network (RNN)")
        # Reshape the data
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        # Reshape the data
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(Xtrain, self.y_train, epochs=10, batch_size=1)
        # predicting the opening prices
        prediction = model.predict(Xtest)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))

        # Plotting
        plt.plot(y_pred, label="Predicted")
        plt.plot(self.y_test, label="Observed")
        plt.legend()

        # Adjust the x-axis for better date handling
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.title("RNN")
        st.pyplot(plt)





# Variables to track which data is being used
flag_fetched = False
flag_uploaded = False

# Logic for "Get data from the net"
if option == "Get data from the net":
    # Sidebar for fetching data
    st.sidebar.subheader('Query parameters')
    start_date = st.sidebar.date_input("Start date", datetime.date(2012, 5, 18))
    end_date = st.sidebar.date_input("End date", datetime.date(2021, 3, 25))

    # Retrieving ticker data
    ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
    tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list)

    # Fetch data using Yahoo Finance
    try:
        data = yf.download(tickerSymbol, start=start_date, end=end_date).reset_index()
        if not data.empty:
            # Process fetched data
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            flag_fetched = True  # Mark fetched data as available
            st.header('**Fetched Stock Data**')
            st.write(data)
        else:
            st.error("No data found for the selected stock and date range.")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")

# Logic for "Upload the data (.csv format)"
elif option == "Upload the data (.csv format)":
    # File uploader
    file = st.file_uploader('Dataset (CSV format)')
    if file is not None:
        try:
            data = pd.read_csv(file)
            flag_uploaded = True  # Mark uploaded data as available
            st.header('**Uploaded Stock Data**')
            st.write(data)
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")

# Ensure only one dataset is being used at a time
if flag_fetched and flag_uploaded:
    st.error("Please select only one method of data input at a time.")
elif not flag_fetched and not flag_uploaded:
    st.warning("Please upload a dataset or select a stock to fetch data.")
elif flag_fetched or flag_uploaded:
    # Only proceed if a valid dataset is available
    st.subheader('Define time window length:')
    st.text('Default is set to 60')
    k = st.number_input('', step=1, min_value=1, value=60, label_visibility="hidden")

    # Create an object of the class and allow user to train the model
    company_stock = stock_predict_DL(data)
    st.subheader('Which Deep Learning model would you like to train?')
    mopt = st.selectbox('', ["Click to select", "LSTM", "MLP", "RNN", "Basic ANN", "Autoencoder"])

    # Call the respective model function based on user's choice
    if mopt == "LSTM":
        company_stock.LSTM_model()
    elif mopt == "MLP":
        company_stock.Mlp_model()
    elif mopt == "RNN":
        company_stock.rnn_model()
    elif mopt == "Autoencoder":
        company_stock.autoen_model()
    elif mopt == "Basic ANN":
        company_stock.basic_ann_model()
