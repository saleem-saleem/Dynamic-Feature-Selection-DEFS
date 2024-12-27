import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(importance_scores, feature_names=None):
    """
    Plot the feature importance scores.

    Args:
        importance_scores (list or numpy array): Feature importance scores.
        feature_names (list, optional): Names of the features. Defaults to indices.
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance_scores))]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_scores, color='skyblue')
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance Visualization")
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    plt.show()

# Example Usage
importance_scores = np.random.rand(10)  # Replace with actual importance scores
feature_names = [f"F{i}" for i in range(10)]
plot_feature_importance(importance_scores, feature_names)
