import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('creditcard.csv')

df['normalized_amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df.drop(['Amount', 'Time'], axis=1, inplace=True)

fraud = df[df['Class'] == 1]
genuine = df[df['Class'] == 0]

genuine_sample = genuine.sample(len(fraud), random_state=42)
balanced_df = pd.concat([fraud, genuine_sample])

X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
