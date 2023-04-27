import pandas as pd


train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)


train.Age = train.Age.fillna(round(train.Age.mean(), 1))
test.Age = test.Age.fillna(round(test.Age.mean(), 1))


train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
