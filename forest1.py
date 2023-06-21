import os
os.chdir("/Users/zach/Desktop/data-use")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix as cm


def convert_k_to_float(value):
    value = value.replace(',', '')
    if 'k' in value:
        return float(value.replace('k', '')) * 1000
    return float(value)

data = pd.read_csv('codereview-questions.csv')
data['Author_Rep'] = data['Author_Rep'].apply(lambda x: convert_k_to_float(x))
X = data[['Author_Rep', 'Question_Score', 'Number_Of_Views', 'Number_Of_Answers', 'Number_Of_Comments', 'Edited']]
y = data['Answer_Accepted']
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)
rfc = RFC(n_estimators=100, random_state=42).fit(X_train, y_train)
train_acc = acc(y_train, rfc.predict(X_train))
test_acc = acc(y_test, rfc.predict(X_test))
train_cm = cm(y_train, rfc.predict(X_train))
test_cm = cm(y_test, rfc.predict(X_test))
imp = rfc.feature_importances_
most_imp = X.columns[np.argmax(imp)]
least_imp = X.columns[np.argmin(imp)]
print(f"train acuracy: {train_acc:.3f}")
print(f"test acuracy: {test_acc:.3f}\n")

print("train confusion matrix:")
print(train_cm)
print("\nTest confusion matrix:")
print(test_cm)

print(f"\nFeature importances:")
for feature, importance in zip(X.columns, imp):
    print(f"{feature}: {importance:.3f}")

print(f"\nmost important feature: {most_imp}")
print(f"least important feature: {least_imp}")
