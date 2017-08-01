"""
import library
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Activation,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
from common import get_block
import sys
# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = pd.read_csv('openprice_ag.csv')
def get_avg(array,days):
	new_array = []
	loops = len(array)-days+1
	for i in range(loops):
		new_array.append(np.mean([array[i],array[i+1],array[i+2]]))
	return new_array

def get_block(array,days):
	new_array = []
	_arr = []
	loops = len(array)-days+1
	if days == 1:
		for i in range(1,loops,days):
			_arr.append([array[i-1],array[i]])
			new_array.append(array[i])
		return new_array,_arr
	elif days == 2:
		for i in range(1,loops,days):
			_arr.append([array[i-1],array[i],array[i+1]])
			new_array.append(np.mean([array[i-1],array[i]]))
			if i <= 20 :
				print i,array[i-1],array[i],np.mean([array[i-1],array[i]])
		return new_array,_arr
	else:
		for i in range(1,loops,days):
			_arr.append([array[i-1],array[i],array[i+1],array[i+2]])
			new_array.append(np.mean([array[i],array[i+1],array[i+2]]))
		return new_array,_arr



days = 2
#new_df = pd.DataFrame(dataframe['datetime'][:-days+1])
#new_df.columns = ['closeprice']
new_df ,_array = get_block(dataframe['price'],days)
_arr_diff = []
for i in _array:
	_arr_diff.append(np.diff(i))

_arr_roi = []
for i in range(len(_arr_diff)):
	tmp_arr = []
	for j in range(len(_arr_diff[i])):
		tmp_arr.append(1+_arr_diff[i][j]/_array[i][j]-8/10000)
	if i<10:
		print _array[i],_arr_diff[i],1+_arr_diff[i][j]/_array[i][j]-8/10000

	_arr_roi.append(tmp_arr)




#print _array_1
#new_df.index = dataframe['datetime'][:-days+1]
new_df = pd.DataFrame(new_df)
new_df.to_csv('new.csv')
dataframe = pd.read_csv('new.csv', usecols=[1], engine='python', skipfooter=0)
dataset1 = dataframe.values
dataset1 = dataset1.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1)
"""
A simple method that we can use is to split the ordered dataset into train and test datasets. The code below
calculates the index of the split point and separates the data into the training datasets with 67% of the
observations that we can use to train our model, leaving the remaining 33% for testing the model.
"""
# split into train and test sets
if len(sys.argv)<2:
	train_size = int(len(dataset1) * 0.4)
else:
	ratio = float(sys.argv[1])
	train_size = int(len(dataset1) * ratio)






#test_size1 = len(dataset1) - train_size
dataset = dataset1
#for i in range(len(test_size1)):

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
	    a = dataset[i:(i+look_back), 0]
	    dataX.append(a)
	    dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1

#
train = dataset[0:train_size,:]
#print "train_data_size: "+str(len(train)), " test_data_size: "+str(len(test))

look_back = 2
trainX, trainY = create_dataset(train, look_back)


#trainY=pd.DataFrame(trainX)
#trainY.to_csv('testx1.csv')
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
#model.add(Dropout(0.5))
model.add(Dense(1))
#model.add(Dense(1, activation='softmax'))

#model.add(Activation('softmax'))
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='mean_squared_error',optimizer='adam')
model.compile(loss='mean_squared_error',optimizer='sgd')

model.fit(trainX, trainY, nb_epoch=10, batch_size=1, verbose=2)

trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])

test_size = len(dataset)-len(train)
n1 = float(sys.argv[2])
n2 = float(sys.argv[3])
#for i in range(5,100):

test = dataset[train_size:len(dataset),:]
test = dataset[train_size+int(n1*test_size):train_size+int(n2*test_size),:]

#test = dataset[0:10:]


testX, testY = create_dataset(test, look_back)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# make predictions

testPredict = model.predict(testX)
#test_prob = model.predict_proba(testX)

# invert predictions
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

diff_testy = np.diff(testY[0])
l_testy = list(map(lambda x :  -1 if x<=0 else 1,diff_testy))

diff_testpredict = np.diff(testPredict[:,0])
l_testpredict = list(map(lambda x : -1 if x<=0.05 else 1,diff_testpredict))
#l_testpredict = list(map(lambda x : -1 if x<=0.00 else 1,diff_testpredict))

#diff_testpredict = np.diff(test_prob[:,0])
#l_testpredict = list(map(lambda x : -1 if x<=0.03 else 1,diff_testpredict))


_arr_roi_1 = _arr_roi[train_size+int(n1*test_size):train_size+int(n2*test_size)]
_array_1 = _array[train_size+int(n1*test_size):train_size+int(n2*test_size)]

#for n in range(50):
#	print np.mean((_array_1[n+1][1],_array_1[n+1][2])),testY[0][n]
#len(_array_1[0])

r1 = pd.DataFrame(testY[0])
r1[1] = testPredict[:,0]

r2 = pd.DataFrame(l_testy)
r2[1] = l_testpredict

diff = np.array(r2[1] - r2[0])
#r1.to_csv('r1.csv')
print '\n'
#r2.to_csv('r2.csv')

roi = []
sum1 = 0
for i in range(len(diff)):
	if l_testpredict[i] == 1:

		
		print testY[0][i],_arr_roi_1[i+look_back],np.product(_arr_roi_1[i+look_back])
		print np.mean([_array_1[i+look_back][0],_array_1[i+look_back][1]]),_array_1[i+look_back]
		roi.append(np.product(_arr_roi_1[i+look_back]))
		sum1 += 1
		print '--------------------'
	else:
		pass
		#print 0,np.mean((_array_1[i+1][1],_array_1[i+1][2])),testY[0][i],_arr_roi_1[i+1],np.product(_arr_roi_1[i+1])
print 'predict roi is %.2f'%(np.product(roi))

#roi = pd.DataFrame(roi)
#roi.to_csv('roi.csv')
roi_perfect = []
sum2 = 0
for i in range(len(diff)):
	if l_testy[i] == 1:
		roi_perfect.append(1+diff_testy[i]/testY[0][i]-16/10000)
		sum2 += 1

print 'test all %.2f'%(sum2/len(l_testpredict))
sum = 0
for i in range(len(diff)):
	if diff[i] == 0:
		sum += 1

print 'Accuracy is %.2f'%(sum/len(l_testpredict))
print '\n'
sum = 0
for i in range(len(diff)):
	if l_testy[i] == l_testpredict[i] == 1:
		sum += 1
if sum1 !=0:
	print 'Precision is %.2f'%(sum/sum1)

print l_testpredict
print len(_arr_roi),len(_arr_diff)
print sum1,sum2,len(l_testpredict)
#print len(l_testpredict),np.sum(np.array(l_testy) - np.array(l_testpredict))





def a():
	1
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
#print(trainY[0])
#print(trainPredict[:,0])
#print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print(testY[0])
#print(testPredict[:,0])
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

