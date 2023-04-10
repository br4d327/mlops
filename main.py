import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from prophet import Prophet



NAME_TRAIN = 'train.csv'
NAME_TEST = 'test.csv'
NAME_PRINT = ''


def data_creation():
    print('Starting data creation')
    print('Checking folders')
    if not os.path.exists('train'):
        os.mkdir('train')

    if not os.path.exists('test'):
        os.mkdir('test')
    
    date_qwe = pd.date_range('2000-05-01', '2010-05-01')
    my_wave = gen_temp(date_qwe)

    cols = {'date': date_qwe, 't': my_wave}


    df = pd.DataFrame(cols)

    qwer = np.random.uniform(-5, 5, df.shape[0])
    df.t = df.t + qwer

    print('train/test split \n 1 year for test')

    df[:-365].to_csv('train/'+NAME_PRINT+NAME_TRAIN, index=False)
    df[-365:].to_csv('test/'+NAME_PRINT+NAME_TEST, index=False)
    print('End save train and test datasets')


def model_preprocessing():
    print('start scaler fit on train data')

    train = pd.read_csv('train/train.csv')
    scaler = RobustScaler()
    train['t'] = scaler.fit_transform(train[['t']])
    train.to_csv('train/train_preprocessing.csv', index=False)

    with open('dump_scaler.pkl', 'wb') as f:
        pickle.dump(scaler,  f)

    print('scaler fitted and saved in main folder')


def model_preparation():
    print('Fitting Prophet')

    df = pd.read_csv('train/train_preprocessing.csv')
    df.columns = ['ds','y']
    model = Prophet()
    model.fit(df)

    with open('model.pkl','wb') as f:
        pickle.dump(model, f)

    print('model saved in main folder')


def model_testing():
    print('evaluate')
    test = pd.read_csv(f'test/test.csv')
    
    with open('dump_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    test['t'] = scaler.transform(test[['t']])
    test.columns = ['ds', 'y']
    test['yhat'] = model.predict(test[['ds']])['yhat']
    score = mean_absolute_error(test['y'], test['yhat'])
    print('MAE score: ', score)


def gen_temp(qwe):
    cycles = 10
    resolution = qwe.shape[0]

    length = np.pi * 2 * cycles

    my_wave = np.sin(np.arange(0, length, (length / resolution)))*20
    return my_wave


if __name__ == '__main__':
    data_creation()
    model_preprocessing()
    model_preparation()
    model_testing()
