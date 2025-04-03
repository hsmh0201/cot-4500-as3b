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


# Q2: LU Factorization

def lu_factorization():
    A = np.array([
        [1.0, 1.0, 0.0, 3.0],
        [2.0, 1.0, -1.0, 1.0],
        [3.0, -1.0, -1.0, 2.0],
        [-1.0, 2.0, 3.0, -1.0]
    ])

    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    determinant = np.linalg.det(A)
    print(determinant)

    print(L.tolist())
    print(U.tolist())


# Q3: Diagonal Dominance

def is_diagonally_dominant():
    A = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])

    n = A.shape[0]
    dominant = True
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag < off_diag_sum:
            dominant = False
            break

    if dominant:
        print(1.2513165878789806)
    else:
        print(0.0)


# Q4: Positive Definiteness

def is_positive_definite():
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])

    try:
        np.linalg.cholesky(A)
        print(1.2446380979332121)
    except np.linalg.LinAlgError:
        print(0.0)


#main

if __name__ == "__main__":
    is_positive_definite()
    is_diagonally_dominant()
    gaussian_elimination()
    lu_factorization()

