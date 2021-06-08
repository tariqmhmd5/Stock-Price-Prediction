import joblib
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM

def scaler(data):
    sc = MinMaxScaler()
    scaled_data = sc.fit_transform(df.values.reshape(-1,1))
    joblib.dump(sc, "scaler.sc")
    return scaled_data

def prepare_data(data,batch_size=20):
    X_Train = []
    Y_Train = []
    for i in range(batch_size, data.shape[0]):

        X_Train.append(data[i-batch_size:i,0])

        Y_Train.append(data[i,0])
        
    # Convert into Numpy Array
    X_Train = np.array(X_Train)
    Y_Train = np.array(Y_Train)
    X_Train = np.reshape(X_Train, newshape=(X_Train.shape[0], X_Train.shape[1], 1))
    return X_Train , Y_Train

def train_model(x,y,ep=50):
    regressor = Sequential()
    regressor.add(LSTM(units = 100, input_shape = (x.shape[1], 1)))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(x,y, epochs=ep, batch_size=1, verbose=2)
    return regressor

    

if __name__=='__main__':
    df = read_csv('TSLA.csv')  #data file reading
    df = df['close']           #selecting close column

    epochs=150                 #setting epochs
    
    train = scaler(df)         #scaling data
    
    x,y = prepare_data(data=train)   #converts data into batches and last element in batch is y
    
    model = train_model(x,y,ep=epochs) #train model
    
    model.save('lstm_model.h5')  # creates a HDF5 file 'my_model.h5'
    print('\n Model saved in present directory!')