import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
	print('evaluate')
	test = pd.read_csv('test/test.csv')

	with open('dump_scaler.pkl', 'rb') as f:
		scaler = pickle.load(f)

	with open('model.pkl', 'rb') as f:
		model = pickle.load(f)

	test['t'] = scaler.transform(test[['t']])
	test.columns = ['ds', 'y']
	test['yhat'] = model.predict(test[['ds']])['yhat']
	score = mean_absolute_error(test['y'], test['yhat'])
	print('MAE score: ', score)
