import sys
import argparse
import numpy as np
import cvxpy as cp
import gurobipy



def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Attempts at the Erdos Similarity Conjecture")

    # Add arguments, all as integers
    parser.add_argument('--n', type=int, required=True, help='n')
    parser.add_argument('--random_seed', type=int, default=1, help='Seed for random number generator')
    parser.add_argument('--lower_bound', type=int, help='Lower bound value')
    parser.add_argument('--enforced_length', type=int, help='Length of constant subsequence')
    parser.add_argument('--enforced_value', type=int, help='Value of constant subsequence')
    # Add verbose as a boolean flag
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')

    # Parse the arguments
    args = parser.parse_args()

    print(f"n = {args.n}", file=sys.stderr)

    if args.lower_bound is not None:
        print("using lower_bound", args.lower_bound, file=sys.stderr)
    else:
        args.lower_bound = 0

    if args.enforced_length is not None:
        assert args.enforced_value is not None
        print(f"enforcing the first {args.enforced_length} to be {args.enforced_value}", file=sys.stderr)
    else:
        assert args.enforced_value is None
        args.enforced_length = 0
        args.enforced_value = 0

    assert args.enforced_value in (0, 1)
    return args


args = parse_arguments()

# Create variables for each argument
n = args.n
random_seed = args.random_seed
lower_bound = args.lower_bound
enforced_length = args.enforced_length
enforced_value = args.enforced_value
verbose = args.verbose


N = 2 ** n


# ChatGPT-4
def create_binary_array(n):
    # Generate a range of numbers from 0 to 2**n - 1
    num_range = np.arange(2 ** n)
    # Convert each number to binary format, pad with zeros to ensure n digits, and convert to a numpy array
    omegas = np.array([list(f"{x:0{n}b}") for x in num_range]).astype(int)
    return omegas


def T_omega(omega):
    ts = [1]
    for a in omega:
        t = (2 * ts[-1] + a) % N
        ts.append(t)
    ts = np.array(ts)
    assert len(ts) == n
    return ts


omegas = create_binary_array(n - 1)


Ts = []
for omega in omegas:
    Ts.append(T_omega(omega))

Ts = np.array(Ts)

T_agg = np.zeros(N)
for T in Ts:
    T_agg[T] += 1

A = cp.Variable(N, boolean=True, name="A")
constraints = []

'''
for x in range(N):
    T_agg_prime = [T_agg[(x + y) % N] for y in range(N)]
    scalar_prod = sum(T_agg_prime[y] * A[y] for y in range(N))
    constraint = scalar_prod >= N // 2
    constraints.append(constraint)

ip = cp.Problem(cp.Minimize(cp.sum(A)), constraints)
ip.solve(solver="GUROBI", verbose=True)
print(ip.value, A.value.astype(int))
exit()
'''

for x in range(N):
    for T in Ts:
        # A + T Minkowski sum covers x.
        # equivalently A intersects T_prime
        T_prime = [(x - y) % N for y in T]
        constraint = sum(A[y] for y in T_prime) >= 1
        constraints.append(constraint)


'''
# Gabor Somlai's (?) extra constraint:
constraint = sum([A[4 * i] for i in range(N // 4)]) == sum([1 for i in range(N // 4)])
constraints.append(constraint)
'''


env = gurobipy.Env()
env.setParam('Seed', int(random_seed))


if enforced_value == 1:
    constraints.append(cp.sum(A[:enforced_length]) == enforced_length)
    # constraints.append(cp.sum(A[enforced_length: 2 * enforced_length - 3]) == 0)
else:
    constraints.append(cp.sum(A[:enforced_length]) == 0)


constraints.append(cp.sum(A) >= lower_bound)
ip = cp.Problem(cp.Minimize(cp.sum(A)), constraints)
ip.solve(solver="GUROBI", verbose=verbose, env=env)



def compact(A):
    lengths = []
    positions = []
    current_length = 0
    for i, value in enumerate(A):
        if value == 1:
            current_length += 1
        else:
            if current_length > 0:
                lengths.append(current_length)
                positions.append(i - current_length)
                current_length = 0
    if current_length > 0:
        lengths.append(current_length)
    lengths = np.array(lengths)
    positions = np.array(positions)
    return lengths, positions


def normalize(A):
    # let's put a zero at the end so that we don't have to bother with wraparound
    for i in range(1, N):
        if (A[i - 1] == 0) and (A[i] == 1):
            break
    A = A[i:].tolist() + A[:i].tolist()
    assert A[0] == 1
    assert A[-1] == 0
    A = np.array(A)
    lengths, positions = compact(A)
    maximal_length = max(lengths)
    start_of_first_longest = None
    for i in range(len(lengths)):
        if lengths[i] == maximal_length:
            start_of_first_longest = positions[i]
            break
    A = A[start_of_first_longest:].tolist() + A[:start_of_first_longest].tolist()
    return np.array(A)


A = A.value.astype(int)
A = normalize(A)

print(f"{int(ip.value)}\t" + " ".join(map(str, A)))

lengths, positions = compact(A)
for length, position in zip(lengths, positions):
    print(f"{position}={length}", end="\t")
print()

# i wanna run it in batch, just collecting many solutions
exit()


A = A.value.astype(int)
for i, a in enumerate(A):
    if a == 1:
        print(i)


unary_Ts = []
for T in Ts:
    unary_T = np.zeros(N, dtype=int)
    unary_T[T] = 1
    unary_Ts.append(unary_T)
unary_Ts = np.array(unary_Ts)

def convolve(A, T):
    mink = T[:, None] * A[None, :]
    convolution = np.zeros(N)
    for i in range(N):
        for j in range(N):
            convolution[(i + j) % N] += mink[i, j]
    return convolution


for unary_T in unary_Ts:
    # print(convolve(A, unary_Ts[0]))
    assert np.all(convolve(A, unary_Ts[0]) >= 1)

print(T_agg)
print(convolve(A, T_agg))
