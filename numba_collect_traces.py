import sys
import argparse
import numpy as np
import cvxpy as cp
import gurobipy
import numba
import matplotlib.pyplot as plt


ALIVE, DEAD = True, False


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
    if a[tail_0] != 0 or a[tail_1] != 0:
        return [(a, tail)], DEAD
    a_0 = a.copy() ; a_0[tail_0] = indx + 1
    a_1 = a.copy() ; a_1[tail_1] = indx + 1
    return [(a_0, tail_0), (a_1, tail_1)], ALIVE


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
    initial = np.zeros(m, dtype=np.int8)
    initial[1] = 1
    current_collection = initial.reshape((1, -1))
    total_collection = []
    iteration = 0
    while len(current_collection) > 0:
        iteration += 1
        next_collection = []
        current_tails = np.argmax(current_collection, axis=1)
        for a, tail in zip(current_collection, current_tails):
            (mini_collection, status) = extend_fast(a, tail)
            if status == DEAD:
                for (b, tail) in mini_collection:
                    if gap_fast(b) >= k:
                        total_collection.append(b)
            else:
                for (b, tail) in mini_collection:
                    # TODO we should add (a!=0).astype(np.int8),
                    # because we don't care about order of introduction.
                    next_collection.append(b)
        next_collection = np.array(next_collection)
        # print("next_collection before unique", len(next_collection))
        _, unique_indices = np.unique(next_collection != 0, axis=0, return_index=True)
        next_collection = next_collection[unique_indices]
        # print("next_collection after unique", len(next_collection))
        print("iteration", iteration, "ongoing", len(next_collection), "harvested", len(total_collection))
        current_collection = next_collection
    total_collection = np.array(total_collection)
    print("total_collection before unique", len(total_collection))
    _, unique_indices = np.unique(total_collection != 0, axis=0, return_index=True)
    total_collection = total_collection[unique_indices]
    print("total_collection after unique", len(total_collection))
    Ts = total_collection
    return Ts


print(len(holey_minimal_sets_fast(33, 10)))


print(len(holey_minimal_sets_fast(83, 26)))
