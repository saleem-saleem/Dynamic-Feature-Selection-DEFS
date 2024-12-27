Dynamic Feature Ensemble Evolution for Enhanced Feature Selection (DE-FS)
=========================================================================
This Python implementation demonstrates the DE-FS algorithm, which integrates multiple feature
selection methods with adaptive and dynamic thresholding to select optimal features for classification.


import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Define Feature Selection Methods
def mutual_information(X, y):
    """Calculates Mutual Information scores for features."""
    return mutual_info_classif(X, y)

def anova_f_test(X, y):
    """Calculates F-statistic (ANOVA) scores for features."""
    return f_classif(X, y)[0]

def correlation_scores(X, y):
    """Calculates correlation coefficients between features and target variable."""
    return np.array([np.corrcoef(X[:, i], y)[0, 1] if np.corrcoef(X[:, i], y)[0, 1] != np.nan else 0 for i in range(X.shape[1])])

# Step 2: Evaluate Feature Subset
def evaluate_feature_subset(X, y, selected_features, model=None):
    """
    Evaluates the performance of a given subset of features using a classification model.
    Default: RandomForestClassifier.
    """
    if model is None:
        model = RandomForestClassifier(random_state=42)

    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Step 3: Dynamic Feature Ensemble Evolution
def dynamic_feature_selection(X, y, max_iterations=10, initial_threshold=0.5):
    """
    Implements the DE-FS algorithm.
    
    Args:
        X (numpy array): Feature matrix.
        y (numpy array): Target variable.
        max_iterations (int): Maximum number of iterations for the algorithm.
        initial_threshold (float): Initial threshold for feature importance.
        
    Returns:
        list: Indices of selected features.
    """
    num_features = X.shape[1]
    threshold = initial_threshold
    best_features = []
    best_score = 0

    feature_selection_methods = [mutual_information, anova_f_test, correlation_scores]

    print("Starting DE-FS Algorithm...")
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")

        # Step 3: Calculate Feature Importance Scores
        importance_scores = np.zeros(num_features)
        for method in feature_selection_methods:
            importance_scores += method(X, y)
        importance_scores /= len(feature_selection_methods)

        # Step 4: Apply Dynamic Thresholding
        selected_features = [i for i, score in enumerate(importance_scores) if score > threshold]
        if not selected_features:
            print("No features exceed the threshold. Stopping early.")
            break

        # Step 5: Evaluate Selected Features
        score = evaluate_feature_subset(X, y, selected_features)
        print(f"Performance with selected features: {score:.4f}")

        # Step 6: Adjust Threshold and Track Best Features
        if score > best_score:
            best_score = score
            best_features = selected_features
            threshold *= 0.9  # Decrease threshold to allow more features
        else:
            threshold *= 1.1  # Increase threshold to be more selective

        print(f"Updated Threshold: {threshold:.4f}")

    print("\nBest Performance Achieved:", best_score)
    print("Optimal Feature Subset Indices:", best_features)
    return best_features

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Step 1: Generate Synthetic Dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)

    # Step 2: Run DE-FS Algorithm
    selected_features = dynamic_feature_selection(X, y, max_iterations=10, initial_threshold=0.5)
    print("\nSelected Features:", selected_features)
