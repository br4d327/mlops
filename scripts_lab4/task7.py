import pandas as pd


train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)


train = train[['Pclass', 'Sex', 'Age']]
test  = test[['Pclass', 'Sex', 'Age']]


train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
