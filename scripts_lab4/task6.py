from catboost.datasets import titanic


titanic()[0].to_csv('data/train.csv')
titanic()[1].to_csv('data/test.csv')
