import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

import os
dataset_path = os.path.join("Data", "Crop_recommendation.csv")
df_crop = pd.read_csv(dataset_path)


# 🔹 Check that the file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

print(f"✅ Dataset found at: {dataset_path}")

# 🔹 Load dataset
df = pd.read_csv(dataset_path)
print(f"Loaded {len(df)} rows of data.")

# 🔹 Train the model
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 🔹 Save the model to your models folder
os.makedirs("models", exist_ok=True)
model_path = "models/RandomForest.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model retrained and saved successfully at: {model_path}")
