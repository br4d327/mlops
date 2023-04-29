import numpy as np
import pandas as pd
import os

NAME_TRAIN = 'train.csv'
NAME_TEST = 'test.csv'
NAME_PRINT = ''


def gen_temp(qwe):
    cycles = 10
    resolution = qwe.shape[0]

    length = np.pi * 2 * cycles

    my_wave = np.sin(np.arange(0, length, (length / resolution)))*20
    return my_wave


if __name__ == '__main__':
    
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
