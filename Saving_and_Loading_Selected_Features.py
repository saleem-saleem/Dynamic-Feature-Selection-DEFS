import json

def save_selected_features(feature_indices, filepath="output/selected_features.json"):
    """Save selected feature indices to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(feature_indices, f)
    print(f"Selected features saved to {filepath}")

def load_selected_features(filepath="output/selected_features.json"):
    """Load selected feature indices from a JSON file."""
    with open(filepath, "r") as f:
        feature_indices = json.load(f)
    print(f"Selected features loaded from {filepath}")
    return feature_indices

# Example Usage
selected_features = [0, 2, 5]  # Replace with actual selected features
save_selected_features(selected_features)
loaded_features = load_selected_features()
print("Loaded Features:", loaded_features)
