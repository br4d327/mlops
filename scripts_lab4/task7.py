import pandas as pd


train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)


df_train = train[['Pclass', 'Sex', 'Age']]
df_test = test[['Pclass', 'Sex', 'Age']]


df_train.to_csv('data/train(PcSexAge).csv')
df_test.to_csv('data/test(PcSexAge).csv')
