from data_ingest import load_data
from preprocessing import preprocess
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("artifacts", exist_ok=True)

def train():
    df = load_data()
    X, y = preprocess(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "artifacts/model.pkl")
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train()