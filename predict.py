import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

def retrieve_files():
    model_file = "models/linear_model.joblib"
    test_file = "models/test_data.joblib"

    if not os.path.exists(model_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Missing files. Please run the training script first.")

    model = joblib.load(model_file)
    X_test, y_test = joblib.load(test_file)

    print(f"Model loaded from: {model_file}")
    return model, X_test, y_test

def run_evaluation(model, X_test, y_test):
    print("Evaluating model...")
    output = model.predict(X_test)

    r2 = r2_score(y_test, output)
    rmse = np.sqrt(mean_squared_error(y_test, output))

    print("Results:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Predictions vs Actuals:")
    for pred, actual in zip(output[:5], y_test[:5]):
        print(f"Predicted: {pred:.3f} | Actual: {actual:.3f}")

def main():
    print("Prediction check started")
    model, X_test, y_test = retrieve_files()
    run_evaluation(model, X_test, y_test)
    print("Prediction check complete")

if __name__ == "__main__":
    main()
