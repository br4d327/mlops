import pandas as pd


train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)


train = train.join(pd.get_dummies(train.Sex))
train.drop(columns='Sex', inplace=True)
test = test.join(pd.get_dummies(test.Sex))
test.drop(columns='Sex', inplace=True)


train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
