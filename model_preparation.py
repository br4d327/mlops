import pandas as pd
from prophet import Prophet
import pickle

if __name__ == '__main__':
    print('Fitting Prophet')

    df = pd.read_csv('train/train_preprocessing.csv')
    df.columns = ['ds','y']
    model = Prophet()
    model.fit(df)

    with open('model.pkl','wb') as f:
        pickle.dump(model, f)

    print('model saved in main folder')
