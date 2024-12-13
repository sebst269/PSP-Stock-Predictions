# Streamlit web app implementation of the project. 

## Pre-requisites :

Make sure to install streamlit if haven't already, to install streamlit use the following command :

```
pip install streamlit
```
All the package requirements along with the versions have been mentioned in the requirements.txt file. 

## How to run?

To run the app, in the anaconda prompt, go to the location where the stock_gui.py file is using the cd command and then run the following line:

```
streamlit run stock_gui.py
```

## Web browser

On execution, a locally hosted page pops up in the browser

You can select between the first option that is to upload the data, or choose to connect to the web and get the data, we also get options to select the company and range of data. 


Suppose we choose to browse and upload the data, we browse for our data file in our file uploader (which has been limited to 200MB data files). 


The head of the data frame is printed along with the options for choosing the period for which we want to train the model to predict a particular day’s price and also a dropdown box for selecting the Deep learning model that we want to train.


The user can choose from the five Deep Learning algorithms – LSTM, MLP, RNN, Basic ANN, and Autoencoder. Suppose the user chooses MLP, the R2 score, Mean Squared Log error, and the output plot with the predicted and observed lines are given. 
