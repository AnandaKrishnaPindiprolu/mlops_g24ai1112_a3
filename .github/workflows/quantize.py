"""
Quantization Script
-------------------
This script loads a trained model and performs quantization to reduce model size.
It includes improved error handling for missing model files.
"""

import os
import joblib

def fetch_model(model_name="linear_model.joblib"):
    """
    Load the trained model file for quantization.
    Searches both the current directory and parent directories if not found.
    """
    print("[INFO] Searching for the trained model...")

    # List of possible locations
    possible_paths = [
        model_name,
        os.path.join("..", model_name),
        os.path.join("..", "..", model_name)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Model found at: {os.path.abspath(path)}")
            return joblib.load(path)

    # If not found in any location
    raise FileNotFoundError(
        f"[ERROR] Model file '{model_name}' not found. "
        f"Please train the model first or copy it to this directory."
    )

def quantize_model(model):
    """
    Dummy quantization step – modify as per your quantization logic.
    Here we just simulate reducing model size.
    """
    print("[INFO] Starting quantization...")
    # Your quantization logic here (example: rounding weights, pruning)
    return model

def save_quantized_model(model, output_name="quantized_model.joblib"):
    """
    Save the quantized model.
    """
    joblib.dump(model, output_name)
    print(f"[INFO] Quantized model saved as: {output_name}")

def main():
    try:
        model = fetch_model()
        q_model = quantize_model(model)
        save_quantized_model(q_model)
        print("[SUCCESS] Quantization completed successfully!")
    except FileNotFoundError as e:
        print(str(e))

if __name__ == "__main__":
    main()