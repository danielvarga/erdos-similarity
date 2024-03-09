import sys
import argparse
import numpy as np
import cvxpy as cp
import gurobipy
import numba
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
import matplotlib.pyplot as plt


ALIVE, DEAD = True, False


@overload(np.all)
def np_all(x, axis=None):

    # ndarray.all with axis arguments for 2D arrays.

    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError("Only accepts NumPy ndarray")

    if not (1 <= x.ndim <= 2):
        raise TypingError("Only supports 1D or 2D NumPy ndarrays")

    if isinstance(axis, types.NoneType):

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 1:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 2:

        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)

        return _np_all_impl

    else:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl


@numba.njit
def numba_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True)
    
    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0
    Returns
    -------
    2D array
        unique rows (or columns) from the input array
    1D array of ints
        indices of unique rows (or columns) in input array
    1D array of ints
        number of instances of each unique row
    """

    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()

    else:
        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind="mergesort")

        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]

    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1

    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx], orig_idx[idx], counts


@numba.njit
def extend_fast(a, tail):
    m = len(a)
    # TODO remove all these slow asserts after confident:
    assert np.argmax(a) == tail
    indx = a[tail]
    assert a.max() == indx
    assert (a !=0 ).sum() == indx

    tail_0 = (2 * tail) % m
    tail_1 = (2 * tail + 1) % m
    aa = np.zeros((1, m), dtype=a.dtype)
    aa[0, :] = a
    if a[tail_0] != 0 or a[tail_1] != 0:
        return aa, DEAD
    aa = np.zeros((2, m), dtype=a.dtype)
    aa[0, :] = a
    aa[1, :] = a
    aa[0, tail_0] = indx + 1
    aa[1, tail_1] = indx + 1
    return aa, ALIVE


# print(extend_fast(np.array([0,1,2,0,0,0,0]), tail=2)) ; exit()


@numba.njit
def rotate_fast(a, k):
    b = np.zeros_like(a)
    b[:k] = a[-k:]
    b[k:] = a[:-k]
    return b


@numba.njit
def gap_fast(a):
    aa = np.concatenate((a != 0, a != 0)).astype(np.int8)
    ones = np.where(aa == 1)[0]
    return (ones[1:] - ones[:-1]).max() - 1


assert gap_fast(np.array([0,0,0,1,0,0,0,1,0])) == 4


# gap must be at least k
@numba.njit
def holey_minimal_sets_fast(m, k):
    MAXIMAL_ONGOING = 1000000
    next_collection = np.empty((MAXIMAL_ONGOING, m), dtype=np.int8)
    initial = np.zeros(m, dtype=np.int8)
    initial[1] = 1
    current_collection = initial.reshape((1, -1))
    total_collection = []
    iteration = 0
    while len(current_collection) > 0:
        iteration += 1
        next_collection = np.empty((MAXIMAL_ONGOING, m), dtype=np.int8)
        next_collection_cursor = 0
        current_tails = np.argmax(current_collection, axis=1)
        for a, tail in zip(current_collection, current_tails):
            (mini_collection, status) = extend_fast(a, tail)
            if status == DEAD:
                for b in mini_collection:
                    if gap_fast(b) >= k:
                        total_collection.append(b)
            else:
                for b in mini_collection:
                    # TODO we should add (b!=0).astype(np.int8),
                    # because we don't care about order of introduction.
                    next_collection[next_collection_cursor] = b
                    next_collection_cursor += 1
                    assert next_collection_cursor <= MAXIMAL_ONGOING
        next_collection = next_collection[:next_collection_cursor]
        # print("next_collection before unique", len(next_collection))
        if len(next_collection) == 0:
            print("no more ongoing")
            break
        _, unique_indices, _ = numba_unique(next_collection != 0, axis=0)
        next_collection = next_collection[unique_indices]
        # print("next_collection after unique", len(next_collection))
        print("iteration", iteration, "ongoing", len(next_collection), "harvested", len(total_collection))
        current_collection = next_collection
    total_collection_2 = np.empty((len(total_collection), m), dtype=np.int8)
    for i in range(len(total_collection)):
        total_collection_2[i] = total_collection[i]
    total_collection = total_collection_2
    print("total_collection before unique", len(total_collection))
    _, unique_indices, _ = numba_unique(total_collection != 0, axis=0)
    total_collection = total_collection[unique_indices]
    print("total_collection after unique", len(total_collection))
    Ts = total_collection
    return Ts


print(len(holey_minimal_sets_fast(33, 10)))


print(len(holey_minimal_sets_fast(83, 26)))
