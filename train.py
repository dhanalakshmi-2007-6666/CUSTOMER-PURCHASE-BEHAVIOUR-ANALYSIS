import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("customer_details.csv")
data = data.dropna(subset=['Satisfaction Level'])
if data['Satisfaction Level'].dtype == 'object':
    data['Satisfaction Level'] = data['Satisfaction Level'].astype(str).str.strip()

    mapping = {
        'Very Unsatisfied': 0,
        'Unsatisfied': 0,
        'Bad': 0,
        'Neutral': 1,
        'Okay': 1,
        'Satisfied': 2,
        'Very Satisfied': 2
    }

    data['Satisfaction Level'] = data['Satisfaction Level'].map(mapping)
data = data.dropna(subset=['Satisfaction Level'])
if 'Discount Applied' in data.columns:
    data['Discount Applied'] = data['Discount Applied'].astype(str).str.strip().map({
        'TRUE': 1, 'True': 1, 'Yes': 1, '1': 1,
        'FALSE': 0, 'False': 0, 'No': 0, '0': 0
    })
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data = data.dropna()

print("Final dataset shape:", data.shape)

X = data.drop(['Customer ID', 'Satisfaction Level'], axis=1, errors='ignore')
y = data['Satisfaction Level']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Model R2 Score:", r2_score(y_test, y_pred))
pickle.dump(model, open("customer_purchase_predict.pkl", "wb"))
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Satisfaction Level")
plt.ylabel("Predicted Satisfaction Level")
plt.title("Actual vs Predicted Customer Satisfaction")

plt.grid(True)
plt.show()
