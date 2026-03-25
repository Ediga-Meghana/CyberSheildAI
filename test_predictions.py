from models.hybrid_model import HybridModel

m = HybridModel()
if not m.load():
    print("Model failed to load.")
    exit(1)

test_cases = [
    "you are stupid",
    "nobody likes you idiot",
    "have a nice day"
]

for t in test_cases:
    res = m.predict(t)
    print(f"Text: '{t}' -> Prediction: {res['prediction']} (Confidence: {res['confidence']}, Category: {res['category']})")
