import sys
import argparse
import numpy as np
import cvxpy as cp
import gurobipy


m, = sys.argv[1:]

m = int(m)


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
iteration = 0
while len(current_collection) > 0:
    iteration += 1
    next_collection = set()
    for a in current_collection:
        mini_collection, status = extend(a, m)
        if status == DEAD:
            for b in mini_collection:
                add_to_collection(total_collection, b)
        else:
            next_collection |= set(mini_collection)
    print("iteration", iteration, "ongoing", len(next_collection), "harvested", len(total_collection), file=sys.stderr)
    current_collection = next_collection


Ts = total_collection

print("final number of sets", len(Ts), file=sys.stderr)


A = cp.Variable(m, boolean=True, name="A")
constraints = []
for x in range(m):
    for T in Ts:
        # A + T Minkowski sum covers x.
        # equivalently A intersects T_prime
        T_prime = [(x - y) % m for y in T]
        constraint = sum(A[y] for y in T_prime) >= 1
        constraints.append(constraint)


def pretty(A):
    return "|" + "".join(map(str, A)).replace("0", ".").replace("1", "X") + "|"


random_seed = 1
verbose = False
env = gurobipy.Env()
env.setParam('Seed', int(random_seed))

ip = cp.Problem(cp.Minimize(cp.sum(A)), constraints)
ip.solve(solver="GUROBI", verbose=verbose, env=env)

A = A.value.astype(int)

print(f"modulus {m}\t|A| {sum(A)}\tA {pretty(A)}")
