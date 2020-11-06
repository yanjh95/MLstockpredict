import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def main():
	data_raw = pd.read_csv("sbux_data.csv",index_col=0) 
	#change this next line depending on number of fearutes
	n_features = 2
	data = data_raw[['close','volume']]
	data = data[::-1]
	# dataset is a np array
	dataset = (data.values)
	training_data_len = len(dataset)

	#split Y and X, because I don't wanna scale Y
	x_train_data = dataset[0:training_data_len , :]
	y_train_data = dataset[0:training_data_len ,0] #<====== change acording to number of features
	print(y_train_data)
	print(x_train_data)
	#Scale the data
	x_scaler = MinMaxScaler(feature_range=(0,1))
	x_train_data = x_scaler.fit_transform(x_train_data)

	x_train = []
	y_train = []
	n_steps = 60
	for i in range(n_steps, training_data_len):
		x_train.append(x_train_data[i-60:i])
		y_train.append(y_train_data[i])
	
	#Convert the x_train and y_train to numpy arrays 
	x_train, y_train = np.array(x_train), np.array(y_train)
	#Reshape the data
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_features))
	'''
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], n_features)))
	model.add(LSTM(50, return_sequences= False))
	model.add(Dense(25))
	model.add(Dense(1))
	#Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error',metrics=[ ['accuracy']])
	#Train the model
	model.fit(x_train, y_train, epochs=100, validation_split=0.2)
	# Creates a HDF5 file 'my_model.h5'
	model_name = 'sbux_npar_model.h5' 
	model.save(model_name)
	print("model saved as:", model_name)

	
	#Get the models predicted price values 
	predictions = model.predict(x_test)
	#predictions = y_scaler.inverse_transform(predictions)

	rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
	print("ROOT MEAN SQUARE ERROR",rmse)
	
	#Visualize the data
	plt.figure(figsize=(14,6))
	plt.title('Model')
	plt.ylabel('Close Price USD ($)', fontsize=10)
	plt.plot(predictions)
	plt.plot((y_test))
	plt.show()
	'''
if __name__ == '__main__':
	main()


