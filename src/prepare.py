import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('data/iris.csv')
train_df, test_df = train_test_split(df, stratify=df['species'], random_state=0)

if not os.path.exists('data/prepared'):
    os.mkdir('data/prepared')

train_df.to_csv('data/prepared/train.csv', index=False)
test_df.to_csv('data/prepared/test.csv', index=False)
