import sys
import argparse
import numpy as np
import cvxpy as cp
import gurobipy


m = 15

ALIVE, DEAD = True, False


def extend(a, m):
    k = a[-1]
    k0 = (2 * k) % m
    k1 = (2 * k + 1) % m
    if k0 in a or k1 in a:
        return [a], DEAD
    if isinstance(a, np.ndarray):
        a = a.tolist()
    elif isinstance(a, tuple):
        a = list(a)
    return [tuple(a + [k0]), tuple(a + [k1])], ALIVE


def rotate(a, k, m):
    a = np.array(a)
    a = np.sort((a + k) % m)
    return tuple(a.tolist())


def add_to_collection(total_collection, a):
    needed = True
    # forall k rotation, forall c in total_collection
    for k in range(m):
        b = frozenset(rotate(a, k, m))
        for c in total_collection:
            if c.issubset(a):
                needed = False
                break
        if not needed:
            break
    if needed:
        total_collection.add(frozenset(a))


current_collection = ((1, ), )
total_collection = set()
while len(current_collection) > 0:
    next_collection = set()
    for a in current_collection:
        mini_collection, status = extend(a, m)
        if status == DEAD:
            for b in mini_collection:
                add_to_collection(total_collection, b)
        else:
            next_collection |= set(mini_collection)
    print("ongoing", len(next_collection), "harvested", len(total_collection))
    current_collection = next_collection


print(len(total_collection))
for a in sorted(total_collection):
    print(list(sorted(a)))
