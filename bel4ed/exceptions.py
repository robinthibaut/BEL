__all__ = [
    "NotFittedError",
    "UndefinedMetricWarning"
]


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    NotFittedError("This instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator."...)

    """


class UndefinedMetricWarning(UserWarning):
    """Warning used when the metric is invalid
    """