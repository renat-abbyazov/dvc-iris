from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import mlflow

from helper import get_git_info

feature_names = ['sepal length (cm)',
                 'sepal width (cm)',
                 'petal length (cm)',
                 'petal width (cm)']

train_df = pd.read_csv('data/prepared/train.csv')
test_df = pd.read_csv('data/prepared/test.csv')

clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_df[feature_names], train_df['species'])

y_pred = clf.predict(test_df[feature_names])
score = accuracy_score(test_df['species'], y_pred)
print(f"accuracy_score = {score}")

dump(clf, 'model.pkl')

git_branch_name, git_origin_url = get_git_info()
mlflow.set_experiment('iris')
with mlflow.start_run():
    tags = {
        'model': 'DecisionTree',
        'git_origin_url': git_origin_url,
    }
    metrics = {'score': score}

    mlflow.set_tags(tags)
    mlflow.log_metrics(metrics)

