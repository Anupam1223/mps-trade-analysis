from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd

def train_classical_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Initializes and trains a suite of classical machine learning models.

    Args:
        X_train: The training feature data.
        y_train: The training target data.

    Returns:
        A dictionary containing the trained model objects.
    """
    print("Training classical models...")
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42) # probability=True for ROC curve
    }

    for name, model in models.items():
        print(f"-> Fitting {name}...")
        model.fit(X_train, y_train)
    
    print("All classical models trained successfully.")
    return models