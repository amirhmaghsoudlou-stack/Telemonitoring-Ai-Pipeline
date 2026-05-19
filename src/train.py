"""Training utilities for the main model (Gradient Boosting)."""

from sklearn.ensemble import GradientBoostingClassifier


def train_gb(X_train, y_train, random_state: int = 42):
    # Train a tree-based Gradient Boosting classifier
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model