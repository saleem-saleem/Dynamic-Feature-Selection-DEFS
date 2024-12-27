from sklearn.model_selection import cross_val_score

def evaluate_with_cross_validation(X, y, selected_features, model=None, cv=5):
    """
    Evaluate feature subset using cross-validation.

    Args:
        X (numpy array): Feature matrix.
        y (numpy array): Target variable.
        selected_features (list): Indices of selected features.
        model (sklearn model, optional): Model to evaluate. Defaults to RandomForestClassifier.
        cv (int): Number of cross-validation folds.

    Returns:
        float: Average cross-validation score.
    """
    if model is None:
        model = RandomForestClassifier(random_state=42)

    X_selected = X[:, selected_features]
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Scores: {scores}")
    return np.mean(scores)

# Example Usage
selected_features = [0, 2, 5]  # Replace with actual selected features
cv_score = evaluate_with_cross_validation(X, y, selected_features)
print("Cross-Validation Average Score:", cv_score)
