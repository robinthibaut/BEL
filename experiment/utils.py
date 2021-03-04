import itertools
import numbers
import os
import shutil
import warnings
from inspect import isclass
from os.path import join as jp
from typing import List

import flopy
import joblib
import numpy as np
import scipy.sparse as sp

from experiment.exceptions import NotFittedError
from experiment.config import Setup

FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def _object_dtype_isnan(X):
    return X != X


def column_or_1d(y, warn=False):
    """Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel().",
                stacklevel=2,
            )
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False"""
    if not accept_large_sparse:
        supported_indices = ["int32"]
        if X.getformat() == "coo":
            index_keys = ["col", "row"]
        elif X.getformat() in ["csr", "csc", "bsr"]:
            index_keys = ["indices", "indptr"]
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if indices_datatype not in supported_indices:
                raise ValueError("Only sparse matrices with 32-bit integer"
                                 " indices are accepted. Got %s indices." %
                                 indices_datatype)


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite, accept_large_sparse):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError("A sparse matrix was passed, but dense "
                        "data is required. Use X.toarray() to "
                        "convert to a dense numpy array.")
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn(
                "Can't check %s sparse matrix for nan or inf." %
                spmatrix.format,
                stacklevel=2,
            )
        else:
            _assert_all_finite(spmatrix.data,
                               allow_nan=force_all_finite == "allow-nan")

    return spmatrix


def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    This utility is meant to be used internally by estimators themselves,
    typically in their own predict / transform methods.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [
            v for v in vars(estimator)
            if v.endswith("_") and not v.startswith("__")
        ]

    if not attrs:
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def _ensure_no_complex_data(array):
    if (hasattr(array, "dtype") and array.dtype is not None
            and hasattr(array.dtype, "kind") and array.dtype.kind == "c"):
        raise ValueError("Complex data not supported\n" "{}\n".format(array))


def _assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""
    # validation is also imported in extmath
    from experiment.algorithms.extmath import _safe_accumulator_op

    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in "fc"
    if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any()
                or not allow_nan and not np.isfinite(X).all()):
            type_err = "infinity" if allow_nan else "NaN, infinity"
            raise ValueError(
                msg_err.format(
                    type_err, msg_dtype if msg_dtype is not None else X.dtype))
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype("object") and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")


def assert_all_finite(X, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix

    allow_nan : bool
    """
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)


def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(
            X,
            ["csr", "csc", "coo"],
            dtype=np.float64,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=False,
        )
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy("F" if X.flags["F_CONTIGUOUS"] else "C") if copy else X
    else:
        if X.dtype.kind in "uib" and X.dtype.itemsize <= 4:
            return_dtype = np.float32
        else:
            return_dtype = np.float64
        return X.astype(return_dtype)


def _is_arraylike(x):
    """Returns whether the input is array-like"""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(
        x, "__array__")


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)


def check_array(
    array,
    accept_sparse=False,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.
        - 'allow-nan': accept only np.nan values in array. Values cannot
          be infinite.

        For object dtyped data, only np.nan is checked and not np.inf.


    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        dtypes_orig = list(array.dtypes)
        # pandas boolean dtype __array__ interface coerces bools to objects
        for i, dtype_iter in enumerate(dtypes_orig):
            if dtype_iter.kind == "b":
                dtypes_orig[i] = np.object

        if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         ". Got {!r} instead".format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(
            array,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            accept_large_sparse=accept_large_sparse,
        )
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                # warnings.simplefilter('error', ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in "iu":
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = np.asarray(array, order=order)
                    if array.dtype.kind == "f":
                        _assert_all_finite(array,
                                           allow_nan=False,
                                           msg_dtype=dtype)
                    array = array.astype(dtype, casting="unsafe", copy=False)
                else:
                    array = np.asarray(array, order=order, dtype=dtype)

            except Exception as e:
                print(e)

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2." %
                             (array.ndim, estimator_name))

        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == "allow-nan")

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required%s." %
                (n_samples, array.shape, ensure_min_samples, context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s." %
                (n_features, array.shape, ensure_min_features, context))

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def data_read(file: str = None, start: int = 0, end: int = None):
    # end must be set to None and NOT -1
    """Reads space-separated dat file"""
    with open(file, "r") as fr:
        lines = np.copy(fr.readlines())[start:end]
        try:
            op = np.array([list(map(float, line.split())) for line in lines],
                          dtype=object)
        except ValueError:
            op = [line.split() for line in lines]
    return op


def folder_reset(folder: str, exceptions: list = None):
    """Deletes files in folder"""
    if not isinstance(exceptions, (list, tuple)):
        exceptions = [exceptions]
    try:
        for filename in os.listdir(folder):
            if filename not in exceptions:
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))
    except FileNotFoundError:
        pass


def empty_figs(root: str):
    """Empties figure folders"""

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            print("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(Setup.Directories.forecasts_dir, root)
    listme = os.listdir(subdir)
    folders = list(
        filter(lambda d: os.path.isdir(os.path.join(subdir, d)), listme))

    for f in folders:
        # pca
        folder_reset(os.path.join(subdir, f, "pca"))
        # cca
        folder_reset(os.path.join(subdir, f, "cca"))
        # uq
        folder_reset(os.path.join(subdir, f, "uq"))
        # data
        folder_reset(os.path.join(subdir, f, "cca"))


def dirmaker(dird: str):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            return 1
    except Exception as e:
        print(e)
        return 0


def load_flow_model(nam_file: str, exe_name: str = "", model_ws: str = ""):
    """
    Loads a modflow model.
    :param nam_file: str: Path to the 'nam' file.
    :param exe_name: str: Path to the modflow exe file.
    :param model_ws: str: Working directory.
    :return:
    """
    flow_loader = flopy.modflow.mf.Modflow.load

    return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)


def load_transport_model(
    nam_file: str,
    modflowmodel,
    exe_name: str = "",
    model_ws: str = "",
    ftl_file: str = "mt3d_link.ftl",
    version: str = "mt3d-usgs",
):
    """
    Loads a transport model.

    :param nam_file: str: Path to the 'nam' file.
    :param modflowmodel: Modflow model object.
    :param exe_name: str: Path to the mt3d exe file.
    :param model_ws: str: Working directory.
    :param ftl_file: str: Path to the 'ftl' file.
    :param version: str: Mt3dms version.
    :return:
    """
    transport_loader = flopy.mt3d.Mt3dms.load
    transport_reloaded = transport_loader(
        f=nam_file,
        version=version,
        modflowmodel=modflowmodel,
        exe_name=exe_name,
        model_ws=model_ws,
    )
    transport_reloaded.ftlfilename = ftl_file

    return transport_reloaded


def remove_sd(res_tree: str):
    """
    Deletes signed distance file out of sub-folders of folder res_tree.
    :param res_tree: str: Path directing to the folder containing the directories of results
    :return:
    """
    for r, d, f in os.walk(res_tree, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if "sd.npy" in f:
            os.remove(jp(r, "sd.npy"))


def remove_incomplete(res_tree: str, crit: str = None):
    """

    :param res_tree: str: Path directing to the folder containing the directories of results.
    :param crit: str: Name of a file according to which delete folder if not present in said folder.
    :return:
    """

    if crit is None:
        ck = np.array([
            os.path.isfile(jp(res_tree, d)) for d in Setup.Files.output_files
        ])
    else:
        ck = np.array([os.path.isfile(jp(res_tree, crit))])

    opt = ck.all()

    if not opt:
        try:
            shutil.rmtree(res_tree)
        except FileNotFoundError:
            pass


def keep_essential(res_dir: str):
    """
    Deletes everything in a simulation folder except specific files.
    :param res_dir: Path to the folder containing results.
    """
    for the_file in os.listdir(res_dir):
        if (not the_file.endswith(".npy") and not the_file.endswith(".py")
                and not the_file.endswith(".xy")
                and not the_file.endswith(".sgems")):

            file_path = os.path.join(res_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def remove_bad_bkt(res_dir: str):
    """
    Loads all breakthrough curves from the results and delete folder in case
    the max computed concentration is > 1.
    :param res_dir: str: Path to result directory.
    """
    bkt_files = []  # Breakthrough curves files
    # r=root, d=directories, f = files
    roots = []
    for r, d, f in os.walk(res_dir, topdown=False):
        # Adds the data files to the lists, which will be loaded later
        if "bkt.npy" in f:
            bkt_files.append(jp(r, "bkt.npy"))
            roots.append(r)
    if bkt_files:
        tpt = list(map(np.load, bkt_files))
        rm = []  # Will contain indices to remove
        for i in range(len(tpt)):
            for j in range(len(tpt[i])):
                # Check results files whose max computed head is > 1 and removes them
                if max(tpt[i][j][:, 1]) > 1:
                    rm.append(i)
                    break
        for index in sorted(rm, reverse=True):
            shutil.rmtree(roots[index])


def data_loader(
    res_dir: str = None,
    roots: List[str] = None,
    test_roots: List[str] = None,
    d: bool = False,
    h: bool = False,
):
    """
    Loads results from main results folder.

    :param roots: Specified roots for training.
    :param test_roots: Specified roots for testing.
    :param res_dir: main directory containing results sub-directories.
    :param d: bool: Flag to load predictor.
    :param h: bool: Flag to load target.
    :return: tp, sd, roots
    """

    # If no res_dir specified, then uses default
    if res_dir is None:
        res_dir = Setup.Directories.hydro_res_dir

    bkt_files = []  # Breakthrough curves files
    sd_files = []  # Signed-distance files
    hk_files = []  # Hydraulic conductivity files
    # r=root, d=directories, f = files

    if roots is None and test_roots is not None:
        if not isinstance(test_roots, (list, tuple)):
            roots = [test_roots]
        else:
            roots = test_roots
    else:
        if not isinstance(roots, (list, tuple)):
            roots: list = [roots]

    [bkt_files.append(jp(res_dir, r, "bkt.npy")) for r in roots]
    [sd_files.append(jp(res_dir, r, "pz.npy")) for r in roots]
    [hk_files.append(jp(res_dir, r, "hk0.npy")) for r in roots]

    if d:
        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
    else:
        tpt = None
    if h:
        sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
    else:
        sd = None

    return tpt, sd, roots


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    """
    cb = [
        list(itertools.combinations(combi, i))
        for i in range(1, combi[-1] + 1)
    ]  # Get all possible wel combinations
    # Flatten and reverse to get all combination at index 0.
    cb = [item for sublist in cb for item in sublist][::-1]
    return cb


def reload_trained_model(root: str, well: str, sample_n: int = 0):
    base_dir = jp(Setup.Directories.forecasts_dir, "base")
    res_dir = jp(Setup.Directories.forecasts_dir, root, well, "obj")

    f_names = list(
        map(lambda fn: jp(res_dir, f"{fn}.pkl"), ["cca", "d_pca", "post"]))
    cca_operator, d_pco, post = list(map(joblib.load, f_names))

    h_pco = joblib.load(jp(base_dir, "h_pca.pkl"))
    h_pred = np.load(jp(base_dir, "roots_whpa", f"{root}.npy"))

    # Inspect transformation between physical and PC space
    dnc0 = d_pco.n_pc_cut
    hnc0 = h_pco.n_pc_cut

    # Cut desired number of PC components
    d_pc_training, d_pc_prediction = d_pco.comp_refresh(dnc0)
    h_pco.test_transform(h_pred, test_root=[root])
    h_pc_training, h_pc_prediction = h_pco.comp_refresh(hnc0)

    # CCA scores
    d_cca_training, h_cca_training = cca_operator.x_scores_, cca_operator.y_scores_

    d, h = d_cca_training.T, h_cca_training.T

    d_obs = d_pc_prediction[sample_n]
    h_obs = h_pc_prediction[sample_n]

    # # Transform to CCA space and transpose
    d_cca_prediction, h_cca_prediction = cca_operator.transform(
        d_obs.reshape(1, -1), h_obs.reshape(1, -1))

    #  Watch out for the transpose operator.
    h2 = h.copy()
    d2 = d.copy()
    tfm1 = post.normalize_h
    h = tfm1.transform(h2.T)
    h = h.T
    h_cca_prediction = tfm1.transform(h_cca_prediction)
    h_cca_prediction = h_cca_prediction.T

    tfm2 = post.normalize_d
    d = tfm2.transform(d2.T)
    d = d.T
    d_cca_prediction = tfm2.transform(d_cca_prediction)
    d_cca_prediction = d_cca_prediction.T

    return d, h, d_cca_prediction, h_cca_prediction, post, cca_operator
