from sklearn.metrics import classification_report
from data_ingest import load_data
from preprocessing import preprocess
import joblib

def evaluate():
    df = load_data()
    X, y = preprocess(df)
    model = joblib.load("artifacts/model.pkl")
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    print(report)

if __name__ == "__main__":
    evaluate()
