import os
import sys

# Ensure correct path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.hybrid_model import HybridModel

def main():
    print("[INIT] Starting Retraining of the Hybrid Model API Pipeline")
    model = HybridModel()
    model.train()
    print("[SUCCESS] Training complete. Models saved to saved_models/")

if __name__ == '__main__':
    main()
