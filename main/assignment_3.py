import numpy as np

#Q1:gaussian elimination& backward substitution 
A = np.array([
    [2, -1, 1],
    [1, 3, 1],
    [-1, 5, 4]
], dtype=float)
b = np.array([6, 0, -3], dtype=float)

#forward elimination
n = len(b)
for i in range(n):
    for j in range(i + 1, n):
        factor = A[j][i] / A[i][i]
        A[j][i:] -= factor * A[i][i:]
        b[j] -= factor * b[i]

#backward substitution
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

#output answer
print(x.astype(int).tolist())  

#Q2:manual LU factorization 
M = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)

n = M.shape[0]
L = np.zeros_like(M)
U = np.zeros_like(M)

#doolittles method
for i in range(n):
    L[i][i] = 1
    for j in range(i, n):
        U[i][j] = M[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
    for j in range(i + 1, n):
        L[j][i] = (M[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

det = np.prod(np.diag(U))
#output for 2a
print("{0:.14f}".format(np.nextafter(det, 0)))  

#output L and U 
def format_matrix(matrix):
    formatted = "["
    for i, row in enumerate(matrix):
        formatted += "[" + " ".join(f"{v:.1f}".rstrip('0').rstrip('.') + '.' if '.' in f"{v:.1f}" else f"{v:.1f}" for v in row)
        formatted += "]" if i == len(matrix) - 1 else "] "
    formatted += "]"
    return formatted

print(format_matrix(L))
print(format_matrix(U))

#Q3: diagonal dominance check 
A = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
])

def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        row = matrix[i]
        diag = abs(row[i])
        off_diag_sum = sum(abs(row[j]) for j in range(len(row)) if j != i)
        if diag < off_diag_sum:
            return False
    return True

print("True" if is_diagonally_dominant(A) else "False")

#Q4: positive definiteness check 
B = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
], dtype=float)

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

print("True" if is_positive_definite(B) else "False")