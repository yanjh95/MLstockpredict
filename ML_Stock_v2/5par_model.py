import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def main():
	data_raw = pd.read_csv("sbux_data.csv",index_col=0) 
	data = data_raw #.filter(['close'])
	data = data[::-1]
	# dataset is a np array
	dataset = (data.values)
	#training_data_len = math.ceil(len(dataset) * .8)
	training_data_len = len(dataset)
	#Scale the data
	x_scaler = MinMaxScaler(feature_range=(0,1))

	#Create the training data set
	#Create the scaled training data set
	x_train_data = dataset[0:training_data_len , :]
	y_train_data = dataset[0:training_data_len ,3]
	x_train_data = x_scaler.fit_transform(x_train_data)

	x_train = []
	y_train = []
	for i in range(60, training_data_len):
		x_train.append(x_train_data[i-60:i])
		y_train.append(y_train_data[i])
	
	#Convert the x_train and y_train to numpy arrays 
	x_train, y_train = np.array(x_train), np.array(y_train)
	#Reshape the data
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
	'''
	#Create the testing data set
	x_test_data = dataset[training_data_len: , :]
	y_test_data = dataset[training_data_len: ,3]
	x_test_data = x_scaler.fit_transform(x_test_data)
	
	#Create the data sets x_test and y_test
	x_test = []
	y_test = []
	for i in range(60, len(dataset) - training_data_len):
		x_test.append(x_test_data[i-60:i])
		y_test.append(y_test_data[i])
	#Convert the data to a numpy array
	x_test ,y_test = np.array(x_test), np.array(y_test)
	#Reshape the data
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5 ))
	'''
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 5)))
	model.add(LSTM(50, return_sequences= False))
	model.add(Dense(25))
	model.add(Dense(1))
	#Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error',metrics=[ ['accuracy', 'mse']])
	#Train the model
	model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2)
	# Creates a HDF5 file 'my_model.h5'
	model_name = 'sbux_badbad_model.h5' 
	model.save(model_name)
	print("model saved as:", model_name)

	'''
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


