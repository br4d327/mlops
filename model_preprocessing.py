import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler


if __name__ == '__main__':
    print('start scaler fit on train data')

    train = pd.read_csv('train/train.csv')
    scaler = RobustScaler()
    train['t'] = scaler.fit_transform(train[['t']])
    train.to_csv('train/train_preprocessing.csv', index=False)

    with open('dump_scaler.pkl', 'wb') as f:
        pickle.dump(scaler,  f)

    print('scaler fitted and saved in main folder')