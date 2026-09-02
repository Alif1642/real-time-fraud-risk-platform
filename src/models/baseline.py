"""Baseline model definitions."""
from sklearn.linear_model import LogisticRegression

from src.config import settings


def logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Balanced sparse Logistic Regression baseline with a practical iteration budget."""
    return LogisticRegression(
        max_iter=settings.logistic_max_iter,
        class_weight="balanced",
        solver="saga",
        random_state=random_state,
        tol=1e-2,
    )
