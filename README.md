# Wafer-Fault-Detection

This project implements an end-to-end MLOps pipeline to detect wafer faults using sensor data. It includes data ingestion, preprocessing, model training, evaluation, and a Streamlit-based web app for real-time predictions

## Project Structure
```bash
├── data_ingest.py          # Loads the wafer dataset
├── preprocessing.py        # Preprocessing logic (imputation, scaling)
├── train.py                # Trains a RandomForestClassifier
├── evaluate.py             # Model evaluation metrics
├── streamlit_app.py        # Streamlit app for predictions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container setup
├── .github/workflows/      # CI/CD GitHub Actions
└── artifacts/model.pkl     # Saved trained model
```
## 🔧 How to Run (Locally with VS Code)

Clone the repository
```bash
git clone <repo_url>
cd Wafer-Fault-Detection
```
(Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
Train the model
```bash
python train.py
```
Run the Streamlit app
```bash
streamlit run streamlit-app.py
```
## 📊 Sample Dataset

Use the provided wafer.csv sample data (simulated) to test the app. It contains 10 sensor features and a binary label (-1 for fault, 1 for good).

## 🚀 Deployment

Using Docker

docker build -t wafer-app .
docker run -d -p 8501:8501 wafer-app

CI/CD (GitHub Actions)

.github/workflows/ci.yml: Trains and builds on push

.github/workflows/cd.yml: Deploys on successful CI run

## 📦 Tech Stack

Python 🐍

Scikit-learn 🔍

Streamlit 📊

Docker 🐳

GitHub Actions 🚀

## 🧠 Model

Algorithm: RandomForestClassifier

Input: Sensor readings

Output: 0 (fault) or 1 (no fault)