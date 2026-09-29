from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

iris = load_iris()

X = iris.data
y = iris.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")

print("Model trained successfully!")
print("model/model.pkl created successfully!")