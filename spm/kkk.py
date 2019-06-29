


import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


plt.switch_backend('TkAgg')  


dates = []
prices = []

def get_data(filename):
	
	
	with open(filename, 'r') as csvfile:
		
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0])) 
			prices.append(float(row[1])) 

	return

def predict_price(dates, prices, x):
	
	dates = np.reshape(dates,(len(dates), 1)) 
	
	
	svr_lin = SVR(kernel= 'linear', C= 1e3) 
	
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 
	
	svr_rbf.fit(dates, prices) 
	svr_lin.fit(dates, prices)
	


	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') 
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') 
	
	plt.xlabel('Date')
	plt.ylabel('Price') 
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]  

get_data('kk.csv') 

predicted_price = predict_price(dates, prices, 29)

print('The predicted prices are:', predicted_price)
