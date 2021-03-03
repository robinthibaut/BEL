__all__ = [
    "NotFittedError",
    "ConvergenceWarning",
    "EfficiencyWarning",
]


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    NotFittedError("This LinearSVC instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator."...)

    """


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    Number of distinct clusters (3) found smaller than n_clusters (4).
    Possibly due to duplicate points in X.
    """


class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.
    """