import joblib
import numpy as np
from pandas import read_csv
from train import prepare_data
from keras.models import load_model

def forecast(X,n):
    reshaped = X[-1].reshape(1,20,1)
    forecast = []
    for i in range(n):
        pred = regressor.predict(reshaped)
        forecast.append(pred[0][0])
        reshaped = np.append(reshaped,pred)[1:].reshape(1,20,1)
    return forecast

if __name__ == '__main__':
    df = read_csv('TSLA.csv')  # Load data to predict contains atleast 20 rows
    df = df['close'].values
    
    no_of_frame_predictions = 5   #change the no. of predictions you want in future 

    scaler = joblib.load('scaler.sc')
    regressor = load_model('lstm_model.h5')

    data = scaler.transform(df.reshape(-1,1))
    
    X,_= prepare_data(data)
    
    forecasts = forecast(X,no_of_frame_predictions)
        
    predictions = scaler.inverse_transform(np.array(forecasts).reshape(-1,1)).flatten()
    print(f'Next {no_of_frame_predictions} Predictions: \n {predictions}')
    input()