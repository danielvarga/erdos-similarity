import sys
import argparse
import numpy as np
import cvxpy as cp
import gurobipy
import matplotlib.pyplot as plt


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


def add_to_collection(total_collection, a, m):
    needed = True
    # forall k rotation, forall c in total_collection
    for k in range(m):
        b = frozenset(rotate(a, k, m))
        for c in total_collection:
            if c.issubset(b):
                needed = False
                break
        if not needed:
            break
    if needed:
        total_collection.add(frozenset(a))


def minimal_sets(m):
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
                    add_to_collection(total_collection, b, m)
            else:
                next_collection |= set(mini_collection)
        # print("iteration", iteration, "ongoing", len(next_collection), "harvested", len(total_collection), file=sys.stderr)
        current_collection = next_collection
    Ts = total_collection
    return Ts


def minkowski(A, B, m):
    A_1 = np.nonzero(A)[0]
    B_1 = np.nonzero(B)[0]
    ms = (A_1[None, :] + B_1[:, None]) % m
    ApB_1 = np.unique(ms.flatten())
    ApB = np.zeros_like(A)
    ApB[ApB_1] = 1
    return ApB


def set_to_vec(T, m):
    v = np.zeros(m, dtype=int)
    v[list(T)] = 1
    return v


def gap(T, m):
    T = np.sort(np.array(list(T)))
    diff = T[1:] - T[:-1]
    last_diff = T[0] + m - T[-1]
    return max((diff.max(), last_diff))


def collect_all_minimals():
    Ts_dict = {}
    for m in range(3, 51, 2):
        Ts = minimal_sets(m)
        Ts_dict[m] = Ts
        gaps = []
        for T in Ts:
            gaps.append(gap(T, m))
        gaps = np.array(gaps)
        if len(gaps) >= 10:
            tenth = np.sort(gaps)[-10]
        else:
            tenth = np.sort(gaps)[0]
        print(f"q={m}\t|Ts|={len(minimal_sets(m))}\ttenth={tenth}\tlargest={gaps.max()}")
        show_sample = False
        if show_sample:
            for T in Ts:
                if gap(T) >= tenth:
                    print(gap(T, m), tuple(T))

    import pickle
    with open("minimal_sets.pkl", "wb") as f:
        pickle.dump(Ts_dict, f)


# collect_all_minimals() ; exit()


def my_hist(data):
    values, counts = np.unique(data, return_counts=True)
    plt.bar(values, counts)
    plt.xticks(values) # Ensure each integer value has its own tick


def my_cumulative(data):
    x_values = np.arange(m + 1)
    cumulative_counts = np.array([np.sum(data <= x - 1e-6) for x in x_values])
    # percentages = 1 - cumulative_counts / len(data)
    plt.plot(x_values / m, len(data) - cumulative_counts, marker='o')
    # plt.xticks(x_values)
    plt.yscale('log')
    plt.title(f"$q = {m}$")
    plt.xlabel("$\\epsilon$")
    plt.ylabel("number of minimal sets with gap at least $\\epsilon$")
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
    # plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())
    # plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1,10)*0.1, numticks=5))



def generate_plots():
    for m in range(3, 67, 2):
        Ts = minimal_sets(m)
        gaps = []
        for T in Ts:
            gaps.append(gap(T, m))
        gaps = np.array(gaps)
        # my_hist(gaps) ; plt.show()
        my_cumulative(gaps)
        print(f"{m}\t{len(Ts)}")
        plt.savefig(f"epsilon-ratios/{m:03}.png")
        plt.clf()


# generate_plots() ; exit()


def verify(A, Ts, m):
    for T_set in Ts:
        T = set_to_vec(T_set, m)
        if not np.all(minkowski(A, T, m) == 1):
            return False
    return True


def pretty(A):
    assert np.all(np.logical_or(A == 0, A == 1))
    return "|" + "".join(map(str, A)).replace("0", ".").replace("1", "X") + "|"



def solve(Ts, prefix, m):
    A = cp.Variable(m, boolean=True, name="A")
    constraints = []
    for x in range(m):
        for T in Ts:
            # A + T Minkowski sum covers x.
            # equivalently A intersects T_prime
            T_prime = [(x - y) % m for y in T]
            constraint = sum(A[y] for y in T_prime) >= 1
            constraints.append(constraint)

    # the first `prefix` must all be 1:
    constraints.append(A[:prefix].sum() == prefix)

    random_seed = 1
    verbose = False
    env = gurobipy.Env()
    env.setParam('Seed', int(random_seed))

    ip = cp.Problem(cp.Minimize(cp.sum(A)), constraints)
    ip.solve(solver="GUROBI", verbose=verbose, env=env)

    A = A.value.astype(int)

    assert verify(A, Ts, m)
    return A


def main():
    m = 61
    epsilon = 0.3
    k = int(epsilon * m)
    Ts = minimal_sets(m)
    print(f"q = {m}")
    print("number of minimal sets", len(Ts), file=sys.stderr)
    A = np.zeros(m, dtype=int)
    A[:k] = 1
    survivors = []
    for T_set in Ts:
        T = set_to_vec(T_set, m)
        if not np.all(minkowski(A, T, m) == 1):
            survivors.append(T_set)
    print(f"number of survivors with epsilon={epsilon} k={k}: {len(survivors)}", file=sys.stderr)
    for T_set in survivors:
        print(len(T_set), tuple(T_set))

    A = solve(survivors, k, m)
    print(f"when using epsilon={epsilon} interval, all minimal sets can be covered: |A| = {A.sum()}, ratio={A.sum() / m}")

    A = solve(survivors, 0, m)
    print(f"set A needed to cover just the {len(survivors)} holey sets: |A| = {A.sum()}, ratio={A.sum() / m}")

main() ; exit()








def find_optimal_interval_pair(Ts):
    for size in range(1, m):
        for a1_end in range(size, 0, -1):
            for a2_start in range(a1_end + 1, m):
                a2_end = a2_start + size - a1_end
                if not 0 <= a2_end <= m:
                    continue
                A = np.zeros(m, dtype=int)
                A[:a1_end] = 1
                A[a2_start: a2_end] = 1
                if verify(A, Ts, m):
                    return A


# A = find_optimal_interval_pair(Ts) ; print(f"modulus {m}\t|A| {sum(A)}\tA {pretty(A)}") ; exit()



print(f"modulus {m}\t|A| {sum(A)}\tA {pretty(A)}")