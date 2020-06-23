from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
train_df, test_df = train_test_split(df, stratify=df['species'], random_state=0)

clf = RandomForestClassifier(random_state=0)
clf.fit(train_df[iris.feature_names], train_df['species'])

y_pred = clf.predict(test_df[iris.feature_names])
print(f"accuracy_score = {accuracy_score(test_df['species'], y_pred)}")

dump(clf, 'model.pkl')
