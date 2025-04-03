import numpy as np


# Q1: Gaussian Elimination

def gaussian_elimination():
    A = np.array([
        [2.0, -1.0, 1.0, 6.0],
        [1.0, 3.0, 1.0, 0.0],
        [-1.0, 5.0, 4.0, -3.0]
    ])

    n = 3

    # Forward elimination
    for i in range(n):
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]
            A[j] = A[j] - ratio * A[i]

    # Backward substitution
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = A[i][n]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    print(f"[ {int(x[0])} {int(x[1])} {int(x[2])} ]")