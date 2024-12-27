from itertools import product

def optimize_hyperparameters(X, y, max_iterations_range, threshold_range):
    """
    Automates hyperparameter tuning for the DE-FS algorithm.

    Args:
        X (numpy array): Feature matrix.
        y (numpy array): Target variable.
        max_iterations_range (list): Range of values for max_iterations.
        threshold_range (list): Range of values for initial_threshold.

    Returns:
        dict: Optimal parameters and their corresponding performance.
    """
    best_score = 0
    best_params = {}

    for max_iter, threshold in product(max_iterations_range, threshold_range):
        print(f"Testing max_iterations={max_iter}, initial_threshold={threshold}")
        selected_features = dynamic_feature_selection(X, y, max_iterations=max_iter, initial_threshold=threshold)
        score = evaluate_feature_subset(X, y, selected_features)
        print(f"Score: {score:.4f}\n")

        if score > best_score:
            best_score = score
            best_params = {"max_iterations": max_iter, "initial_threshold": threshold}

    return {"best_score": best_score, "best_params": best_params}

# Example Usage
best_config = optimize_hyperparameters(
    X, y, max_iterations_range=[5, 10, 15], threshold_range=[0.3, 0.5, 0.7]
)
print("Optimal Configuration:", best_config)
