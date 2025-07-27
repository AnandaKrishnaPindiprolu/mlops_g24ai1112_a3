import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def load_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    print("Data loaded")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print("Feature labels:", data.feature_names)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_output = model.predict(X_train)
    test_output = model.predict(X_test)
    results = {
        "Train_R2": r2_score(y_train, train_output),
        "Test_R2": r2_score(y_test, test_output),
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, train_output)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, test_output)),
    }
    print("Performance:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    return model, (X_test, y_test)

def save_files(model, test_data):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_model.joblib")
    joblib.dump(test_data, "models/test_data.joblib")
    print("Artifacts stored")
    print("Intercept:", model.intercept_)
    print("Coefficients shape:", model.coef_.shape)
    print("First 5 Coefficients:", model.coef_[:5])

def main():
    print("Training initiated")
    X, y = load_data()
    model, test_data = train_model(X, y)
    save_files(model, test_data)
    print("Training complete")

if __name__ == "__main__":
    main()
