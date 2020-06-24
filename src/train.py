from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd


feature_names = ['sepal length (cm)',
                 'sepal width (cm)',
                 'petal length (cm)',
                 'petal width (cm)']

train_df = pd.read_csv('data/prepared/train.csv')
test_df = pd.read_csv('data/prepared/test.csv')

clf = RandomForestClassifier(random_state=0)
clf.fit(train_df[feature_names], train_df['species'])

y_pred = clf.predict(test_df[feature_names])
print(f"accuracy_score = {accuracy_score(test_df['species'], y_pred)}")

dump(clf, 'model.pkl')
