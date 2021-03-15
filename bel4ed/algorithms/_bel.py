from ._pfa import _PFA

__all__ = ["BEL"]


class BEL(_PFA):
    """Bayesian Evidential Learning (BEL)."""

    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            norm_y_weights=True,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
