from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess(df):
    X = df.drop('label', axis=1)
    y = df['label'].map({-1: 0, 1: 1})  # binary format

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y