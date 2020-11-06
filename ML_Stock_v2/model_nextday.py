# load and evaluate a saved model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
 
# load model
model = load_model('sbux_bad_model.h5')
# load dataset
model.summary()
data = pd.read_csv("sbux_100x5data.csv",index_col=0) 
data = data[::-1]
for i in range(10):
	x = (data[40-i:100-i])
	x_scaler = MinMaxScaler(feature_range=(0,1))
	x = x_scaler.fit_transform(x)
	x = np.reshape(x, (1, 60, 5))
	print(model.predict(x))

