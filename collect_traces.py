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


def gap(T, m):
    T = np.sort(np.array(list(T)))
    diff = T[1:] - T[:-1]
    last_diff = T[0] + m - T[-1]
    return max((diff.max(), last_diff))


def gap_position(T, m):
    T = np.sort(np.array(list(T)))
    diff = T[1:] - T[:-1]
    last_diff = T[0] + m - T[-1]
    diff = np.append(diff, [last_diff])
    gp_index = np.argmax(diff)
    return T[gp_index], diff[gp_index]


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


# gap must be at least k
def holey_minimal_sets(m, k):
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
                    if gap(b, m) >= k:
                        add_to_collection(total_collection, b, m)
            else:
                for b in mini_collection:
                    if gap(b, m) >= k:
                        next_collection.add(b)
        # print("iteration", iteration, "ongoing", len(next_collection), "harvested", len(total_collection), file=sys.stderr)
        current_collection = next_collection
    Ts = total_collection
    return Ts


def pretty(A):
    assert np.all(np.logical_or(A == 0, A == 1))
    return "|" + "".join(map(str, A)).replace("0", ".").replace("1", "X") + "|"


def pretty_set(T, m):
    return pretty(set_to_vec(T, m))


def test_holey_minimal_sets():
    # m = 501 ; k = int(m * 0.45)
    m = 71 ; k = int(m * 0.4)
    for T in holey_minimal_sets(m, k):
        print(pretty_set(T, m))
    print(m, k, len(holey_minimal_sets(m, k)))


# test_holey_minimal_sets() ; exit()


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
    # env.setParam('Seed', int(random_seed))

    ip = cp.Problem(cp.Minimize(cp.sum(A)), constraints)
    ip.solve(solver="GUROBI", verbose=verbose, env=env)

    A = A.value.astype(int)

    assert verify(A, Ts, m)
    return A



def find_survivors(Ts, m, k):
    A = np.zeros(m, dtype=int)
    A[:k] = 1
    survivors = []
    for T_set in Ts:
        T = set_to_vec(T_set, m)
        if not np.all(minkowski(A, T, m) == 1):
            survivors.append(T_set)
    return survivors



# TODO that's a super dumb way to construct these,
# the proper way is repeated halving starting from m+start.
def easy_minimal_sets_starting_at(m, start):
    current_collection = [[start]]
    total_collection = []
    iteration = 0
    while len(current_collection) > 0:
        iteration += 1
        next_collection = []
        for a in current_collection:
            tail = a[-1]
            t0 = 2 * tail
            t1 = 2 * tail + 1
            for next_tail in (2 * tail, 2 * tail + 1):
                if next_tail < m:
                    next_collection.append(a + [next_tail])
                elif next_tail % m == start:
                    total_collection.append(a)
        current_collection = next_collection
        # print(iteration, len(current_collection), len(total_collection))
    return total_collection


# these start somewhere not necessarily at 1, double until
# they wrap around, and then they have to exactly hit their first element
def easy_minimal_sets(m):
    total_collection = []
    for start in range(1, m):
        collection = easy_minimal_sets_starting_at(m, start)
        total_collection += collection
    return total_collection


for m in range(11, 131, 2):
    survivors = easy_minimal_sets_starting_at(m, 1)
    # for T in survivors: print(T)
    A = solve(survivors, 0, m)
    print(f"{m}\t{sum(A)}\t{pretty(A)}")
    

exit()


def test_easy_minimal_sets():
    m = 61
    lg = -1
    k = 1
    while k < m:
        k *= 2
        lg += 1
    print(f"kinda log_2({m}) = {lg}")
    for start in range(1, m, 10):
        lb = start * 2 ** lg
        ub = (start + 1) * 2 ** lg - 1
        print(start, lb, ub)


easy_minimal_sets() ; exit()


def iterative_narrowing(m):
    last_set_count = 0
    for k in range(m, 1, -1):
        survivors = holey_minimal_sets(m, k)
        set_count = len(survivors)
        if set_count == 0:
            continue


        # things are starting to slow down too much:
        if len(survivors) > 30:
            break

        if set_count == last_set_count:
            # we already have this information
            continue
        last_set_count = set_count
        print(m, k, set_count)
        for T in survivors:
            print(list(T))

        A = solve(survivors, 0, m)
        print(f"set A needed to cover just the {len(survivors)} holey sets: |A| = {A.sum()}, ratio={A.sum() / m}")


iterative_narrowing(87) ; exit()


def main():
    m = 61
    epsilon = 0.3
    k = int(epsilon * m)
    print(f"q = {m}")
    do_it_smart = False
    if do_it_smart:
        survivors = holey_minimal_sets(m, k)
        print(f"number of holey minimals with epsilon={epsilon} k={k}: {len(survivors)}")
    else:
        Ts = minimal_sets(m)
        survivors = find_survivors(Ts, m, k)
        print("number of minimal sets", len(Ts))
        print(f"number of holey survivors with epsilon={epsilon} k={k}: {len(survivors)}")

    for T_set in survivors:
        print(len(T_set), tuple(sorted(T_set)), f"{gap(T_set, m) / m :.4f}")

    A = solve(survivors, k, m)
    print(f"when using epsilon={epsilon} interval, all minimal sets can be covered: |A| = {A.sum()}, ratio={A.sum() / m}")

    A = solve(survivors, 0, m)
    print(f"set A needed to cover just the {len(survivors)} holey sets: |A| = {A.sum()}, ratio={A.sum() / m}")


# main() ; exit()



def gap_position_of_uglies():
    epsilon = 0.3
    for m in [21]:
        print("===")
        print(f"q = {m}")
        k = int(epsilon * m)
        survivors = holey_minimal_sets(m, k)
        for T in survivors:
            position, gap_size = gap_position(T, m)
            print(position / m, gap_size / m, list(sorted(T)))


# gap_position_of_uglies() ; exit()


def size_of_uglies():
    epsilon = 0.2
    for m in [67, 71]: # range(11, 81, 2):
        k = int(epsilon * m)
        survivors = holey_minimal_sets(m, k)
        sizes = np.array(sorted([len(T) for T in survivors]))
        print(m, sizes)
        plt.hist(sizes)
        plt.show()


size_of_uglies() ; exit()


def lower_bound_loop():
    epsilon = 0.3
    for m in range(3, 200, 2):
        k = int(epsilon * m)
        survivors = holey_minimal_sets(m, k)

        # print(f"{m}\t{len(survivors)}") ; continue

        A_upper = solve(survivors, k, m)
        ub = A_upper.sum()
        A_lower = solve(survivors, 0, m)
        lb = A_lower.sum()
        print(f"{m}\t{len(survivors)}\t{lb}\t{ub}")
        print(m, "lower", pretty(A_lower))
        print(m, "upper", pretty(A_upper))


lower_bound_loop() ; exit()
