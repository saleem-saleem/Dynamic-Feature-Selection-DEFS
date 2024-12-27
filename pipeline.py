def run_pipeline(dataset_path, max_iterations=10, initial_threshold=0.5):
    """
    End-to-end pipeline for preprocessing, DE-FS, and evaluation.

    Args:
        dataset_path (str): Path to the dataset CSV file.
        max_iterations (int): Number of iterations for DE-FS.
        initial_threshold (float): Initial threshold for feature selection.
    """
    import pandas as pd

    # Step 1: Load Dataset
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target

    # Step 2: Run DE-FS Algorithm
    selected_features = dynamic_feature_selection(X, y, max_iterations, initial_threshold)
    print("Selected Features:", selected_features)

    # Step 3: Evaluate with Cross-Validation
    cv_score = evaluate_with_cross_validation(X, y, selected_features)
    print(f"Cross-Validation Score: {cv_score:.4f}")

# Example Usage
run_pipeline("data/student_performance.csv", max_iterations=10, initial_threshold=0.5)
