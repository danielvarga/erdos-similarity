import sys
import numpy as np


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


def altern(lengths, positions, N):
    ls = []
    last = 0
    for length, position in zip(lengths, positions):
        if position != 0:
            ls.append(position - last)
        ls.append(length)
        last = position + length
    assert last != N
    ls.append(N - last)
    assert len(ls) % 2 == 0
    ls = np.array(ls).reshape((-1, 2))
    return ls


def longest_prefix(arr):
    if arr[0] == 0:
        return 0
    elif np.all(arr == 1):
        return len(arr)
    return np.argmax(arr == 0)


As = []
for l in sys.stdin:
    A = np.array(list(map(int, l.split())))
    As.append(A)

As = sorted(As, key=lambda A: - longest_prefix(A))
As = np.array(As)
N = As.shape[-1]


def pretty(A):
    return "|" + "".join(map(str, A)).replace("0", ".").replace("1", "X") + "|"

for A in As:
    lengths, positions = compact(A)
    alt = altern(lengths, positions, N)
    violation = (alt[:, 1] - alt[:, 0]).min()
    if violation < -1:
        print(pretty(A))
        # print(list(zip(lengths, positions)))
        print(alt.T)
        print(alt[:, 1] - alt[:, 0])
        print("-----")
